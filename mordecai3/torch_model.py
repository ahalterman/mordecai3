## Read in the BERT embedding for each place name
## and predict the country using pytorch
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from pandas import read_csv
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def convert_to_numpy(tensor):
    """Convert a tensor to numpy array, handling both CuPy and NumPy inputs."""
    if hasattr(tensor, 'get'):  # CuPy array
        return tensor.get()
    else:  # Already a NumPy array or other array-like
        return np.asarray(tensor)
    
# Extra per-candidate features carried by the `_enriched` pickles that
# tools/enrich_pickles.py writes. Blocks are enabled by name; the order here is
# the order the columns are appended to gaz_info, so a given --feature-blocks
# string always produces the same layout.
FEATURE_BLOCKS = {
    "prom": ["log_population", "has_population", "is_max_pop", "log_pop_rel",
             "is_max_pop_exact_match"],
    "name": ["exact_name_match", "exact_altname_match"],
    "cue": ["mention_admin_cue", "is_admin_class"],
    "sib": ["sib_adm1", "sib_adm2", "sib_country", "ap_twin"],
    "geo": ["log_min_km_anchor", "anchor_same_adm1_frac", "frac_anchors_150km",
            "frac_anchors_50km", "log_mean_sibmin", "frac_sibs_50km",
            "anchor_same_country_frac", "is_parent_of_anchor",
            "frac_anchors_inside"],
    "shape": ["log_n_same_name", "log_n_exact_matches", "is_unique_exact_match",
              "is_seat_any"],
    "cf": ["min_dist_cf", "max_dist_cf", "avg_dist_cf", "ascii_dist_cf"],
    # Round 5 of the enrichment: the A/P twin test rerun on the
    # admin-word-stripped name key ("Yangon Region" = "Yangon") at both the
    # 0.15-degree gate and a 0.5-degree one (a strict superset, which catches
    # province-vs-city pairs the narrow gate drops), and a flag for
    # historical gazetteer rows (*H feature codes, e.g. Kathmandu District
    # ADM3H). Appended last so every earlier column keeps its index and caches
    # built before this stay readable for recipes that do not ask for it.
    "strip": ["ap_twin_stripped", "ap_twin_stripped_wide", "is_historical"],
}

# What a padding row gets: the bad/neutral end of each feature, mirroring the
# sentinels enrich_pickles gives its own NULL placeholder row. 0.0 reads as "no
# evidence" for an indicator or a fraction, but it is the *best* value for a
# distance, so the two log-distances take the no-anchor sentinel and the
# case-folded edit distances take the worst normalised distance.
_PAD_SENTINEL = {"log_min_km_anchor": 4.301051709845226,
                 "log_mean_sibmin": 4.301051709845226,
                 "min_dist_cf": 1.0, "max_dist_cf": 1.0,
                 "avg_dist_cf": 1.0, "ascii_dist_cf": 1.0}


# The nine original gazetteer features, in the order create_gaz_features has
# always stacked them, and the padding value each one uses.
GAZ_BASE_KEYS = ["alt_name_length", "max_dist", "avg_dist", "min_dist",
                 "ascii_dist", "adm1_count", "country_count",
                 "admin1_parent_match", "country_code_parent_match"]
_BASE_PAD = {"alt_name_length": 99, "max_dist": 99, "avg_dist": 99,
             "min_dist": 99, "ascii_dist": 99}

# Every feature a candidate can contribute, in a fixed order. `compact` builds
# one float32 matrix per entity in exactly this order (see tools/train.py).
ALL_FEATURE_KEYS = GAZ_BASE_KEYS + [k for keys in FEATURE_BLOCKS.values()
                                    for k in keys]


def pad_value(key):
    """What a padding row carries for one feature."""
    if key in _BASE_PAD:
        return _BASE_PAD[key]
    return _PAD_SENTINEL.get(key, 0.0)


def expand_feature_blocks(blocks):
    """Block names -> the flat, ordered list of candidate keys they name."""
    if not blocks:
        return []
    if isinstance(blocks, str):
        blocks = [i.strip() for i in blocks.split(",") if i.strip()]
    unknown = [b for b in blocks if b not in FEATURE_BLOCKS]
    if unknown:
        raise ValueError(f"Unknown feature block(s) {unknown}; "
                         f"known: {sorted(FEATURE_BLOCKS)}")
    keys = []
    for name in FEATURE_BLOCKS:      # canonical order, not the order given
        if name in blocks:
            keys.extend(FEATURE_BLOCKS[name])
    return keys


class ProductionData(Dataset):
    _country_dict = None
    _feature_code_dict = None

    def __init__(self, es_data, max_choices=25, max_codes=50,
                 oov_bucket_fix=False, feature_blocks=None,
                 full_null_row=False):
        self.max_choices = max_choices
        self.max_codes = max_codes
        # Off by default: index 52 ("NULL") stays the out-of-vocabulary feature
        # code bucket and the reserved last row keeps a real candidate's country.
        self.oov_bucket_fix = oov_bucket_fix
        # Empty by default, so gaz_info keeps its original 9 columns.
        self.extra_keys = expand_feature_blocks(feature_blocks)
        self.full_null_row = full_null_row
        self.country_dict = self._get_country_dict()
        self.feature_code_dict = self._get_feature_code_dict()
        self.placename_tensor = np.array([
            convert_to_numpy(i['tensor']) for i in es_data
        ]).astype(np.float32)
        self.doc_tensor = np.array([
            convert_to_numpy(i['doc_tensor']) for i in es_data
        ]).astype(np.float32)
        self.other_locs_tensor = np.array([
            convert_to_numpy(i['locs_tensor']) for i in es_data
        ]).astype(np.float32)
        self.feature_codes = self.create_feature_codes(es_data)
        self.country_codes = self.create_country_codes(es_data)
        self.gaz_info = self.create_gaz_features(es_data).astype(np.float32)
        #self.gaz_info[n][-1] = np.array([0] * 9)
        self.mask = self.create_mask(es_data)

    def create_mask(self, es_data):
        """1.0 for rows that hold a real candidate, 0.0 for padding.

        Roughly half of the 500 rows are padding on an average example. The
        mask is always computed and always handed to the model; only a model
        built with mask_padding=True acts on it.

        The last row is the reserved "no correct answer" slot -- it is a
        trained class (`create_labels` points there when nothing is correct),
        so it stays live even when it is past the end of the candidate list.
        """
        mask = np.zeros((len(es_data), self.max_choices), dtype=np.float32)
        for n, ex in enumerate(es_data):
            live = min(len(ex['es_choices']), self.max_choices)
            mask[n, :live] = 1.0
        mask[:, -1] = 1.0
        return mask
        
    def choices_window(self, ex):
        """The candidates that fill the max_choices rows of one example.

        When a mention has at least max_choices hits, the plain slice keeps
        max_choices real candidates and then the row-[-1] sentinels overwrite
        the last one -- so a real candidate is silently destroyed and the true
        "no correct answer" row, which the gazetteer list ends with, is dropped
        (audit item B2). Under full_null_row the slice leaves room for it.
        """
        ch = ex['es_choices']
        if self.full_null_row and len(ch) > self.max_choices:
            return ch[0:self.max_choices - 1] + [ch[-1]]
        return ch[0:self.max_choices]

    def __getitem__(self, index):
        return {"placename_tensor": self.placename_tensor[index],  
                "doc_tensor": self.doc_tensor[index], 
                "other_locs_tensor": self.other_locs_tensor[index],
                "feature_codes": self.feature_codes[index], 
                "country_codes": self.country_codes[index],
                "gaz_info": self.gaz_info[index],
                "mask": self.mask[index]}
        
    
    def __len__ (self):
        return len(self.placename_tensor)

    # need to make this into a one-hot matrix, not a vector.
    # Inside the model, it should be a 3d one hot tensor, not binary.
    def create_feature_codes(self, es_data):
        all_feature_codes = []
        for ex in es_data:
            feature_code_raw = [i['feature_code'] for i in self.choices_window(ex)]
            feature_code_raw += ['NULL'] * (self.max_choices - len(feature_code_raw))
            feature_code_raw = feature_code_raw[0:self.max_choices]
            ## Pytorch embedding layers need indices, not one-hot
            # len(dict)+1 == 52, which is already the index of "NULL": every
            # out-of-vocabulary code was being folded into the NULL bucket.
            # 51 is unused, so under the flag OOV gets its own embedding.
            oov = len(self.feature_code_dict) if self.oov_bucket_fix else len(self.feature_code_dict)+1
            feature_codes = [self.feature_code_dict[i] if i in self.feature_code_dict else oov for i in feature_code_raw]
            # the last one is an other/not present category
            feature_codes[-1] = 53
            feature_codes = np.array(feature_codes, dtype="int")
            all_feature_codes.append(feature_codes)
        all_feature_codes = np.array(all_feature_codes).astype(np.int32)
        return all_feature_codes

    def create_country_codes(self, es_data):
        all_country_codes = []
        for ex in es_data:
            country_code_raw = [i['country_code3'] for i in self.choices_window(ex)]
            country_code_raw += ['NULL'] * (self.max_choices - len(country_code_raw))
            country_code_raw = country_code_raw[0:self.max_choices]
            country_codes = [self.country_dict[i] for i in country_code_raw]
            # The last row is the reserved "no answer" slot: its feature code
            # and gazetteer features are overridden below/above, but its
            # country was left as whatever candidate happened to land there.
            if self.oov_bucket_fix:
                country_codes[-1] = self.country_dict["NULL"]
            country_codes = np.array(country_codes, dtype="int")
            all_country_codes.append(country_codes)
        all_country_codes = np.array(all_country_codes).astype(np.int32)
        return all_country_codes

    def create_gaz_features(self, es_data):
        """
        Format all non-query/gazetteer-only features.

        Specifically, this includes edit distance features and adm1 and country overlap
        """
        if es_data and 'feat_matrix' in es_data[0]:
            return self._gaz_from_matrix(es_data)
        edit_info = []
        for ex in es_data:
            window = self.choices_window(ex)
            alt_name_length = [i['alt_name_length'] for i in window]
            alt_name_length += [99] * (self.max_choices - len(alt_name_length))
            min_dist = [i['min_dist'] for i in window]
            min_dist += [99] * (self.max_choices - len(min_dist))
            max_dist = [i['max_dist'] for i in window]
            max_dist += [99] * (self.max_choices - len(max_dist))
            avg_dist = [i['avg_dist'] for i in window]
            avg_dist += [99] * (self.max_choices - len(avg_dist))
            ascii_dist = [i['ascii_dist'] for i in window]
            ascii_dist += [99] * (self.max_choices - len(ascii_dist))
            adm1_overlap = [i['adm1_count'] for i in window]
            adm1_overlap += [0] * (self.max_choices - len(adm1_overlap))
            country_overlap = [i['country_count'] for i in window]
            country_overlap += [0] * (self.max_choices - len(country_overlap))
            in_adm1 = [i['admin1_parent_match'] for i in window]
            in_adm1 += [0] * (self.max_choices - len(in_adm1))
            in_country = [i['country_code_parent_match'] for i in window]
            in_country += [0] * (self.max_choices - len(in_country))
            #es_position = normalize(es_position)
            alt_name_length[-1] = -1 
            max_dist[-1] = -1
            avg_dist[-1] = -1 
            min_dist[-1] = -1 
            ascii_dist[-1] = -1 
            adm1_overlap[-1] = -1 
            country_overlap[-1] = -1
            in_adm1[-1] = -1
            in_country[-1] = -1
            cols = [alt_name_length, max_dist, avg_dist, min_dist,
                    ascii_dist, adm1_overlap, country_overlap, in_adm1, in_country]
            for key in self.extra_keys:
                vals = [i[key] for i in window]
                vals += [_PAD_SENTINEL.get(key, 0.0)] * (self.max_choices - len(vals))
                # Same convention the nine original features use for the
                # reserved "no correct answer" row: a constant the scorer can
                # learn a fixed bias for.
                vals[-1] = -1
                cols.append(vals)
            ed = np.transpose(np.array(cols))
            edit_info.append(ed)
        ed_stack = np.stack(edit_info)
        return ed_stack

    def _gaz_from_matrix(self, es_data):
        """Same array, read from the per-entity matrix `compact_candidates` built.

        Holding 500 candidate dicts per entity costs ~25 GB on this corpus, so
        the loader collapses the numeric features into one float32 matrix per
        entity and slims the dicts down to what the scorer reports on. The
        values, their order, the padding and the sentinel row are identical to
        the dict path above -- `tools/train.py` verifies that by rerunning a
        seed and diffing the metrics.
        """
        keys = GAZ_BASE_KEYS + self.extra_keys
        idx = np.array([ALL_FEATURE_KEYS.index(k) for k in keys])
        pad = np.array([pad_value(k) for k in keys], dtype=np.float32)
        width = es_data[0]['feat_matrix'].shape[1]
        if idx.size and idx.max() >= width:
            missing = [k for k in keys if ALL_FEATURE_KEYS.index(k) >= width]
            raise ValueError(
                f"feat_matrix has {width} columns but {missing} live past that: "
                "the compacted cache predates these features. Delete the "
                "*_compact.pkl files and rerun `train.py compact-cache`.")
        out = np.empty((len(es_data), self.max_choices, len(keys)), dtype=np.float32)
        out[:] = pad
        for n, ex in enumerate(es_data):
            fm = ex['feat_matrix']
            if self.full_null_row and fm.shape[0] > self.max_choices:
                fm = np.vstack([fm[0:self.max_choices - 1], fm[-1:]])
            live = min(fm.shape[0], self.max_choices)
            out[n, :live] = fm[:live][:, idx]
            out[n, -1] = -1
        return out

    @classmethod
    def _get_country_dict(cls):
        if cls._country_dict is None:
            pt = os.path.dirname(os.path.realpath(__file__))
            fn = os.path.join(pt, "assets", "wikipedia-iso-country-codes.txt")
            country = read_csv(fn)
            country_dict = {i:n for n, i in enumerate(country['Alpha-3 code'].to_list())}
            country_dict["CUW"] = len(country_dict)
            country_dict["XKX"] = len(country_dict)
            country_dict["SCG"] = len(country_dict)
            country_dict["SSD"] = len(country_dict)
            country_dict["BES"] = len(country_dict)
            country_dict["SXM"] = len(country_dict)
            country_dict["NULL"] = len(country_dict)
            country_dict["NA"] = len(country_dict)
            cls._country_dict = country_dict
        return cls._country_dict

    @classmethod
    def _get_feature_code_dict(cls):
        if cls._feature_code_dict is None:
            pt = os.path.dirname(os.path.realpath(__file__))
            fn = os.path.join(pt, "assets", "feature_code_dict.json")
            with open(fn, "r") as f:
                cls._feature_code_dict = json.load(f)
        return cls._feature_code_dict


# The nine Geonames feature classes, plus a bucket for the NULL placeholder and
# the handful of candidates with no class at all. Coarser than the 53 feature
# codes and far better populated, which is what makes it usable as an auxiliary
# target.
FEATURE_CLASSES = ["A", "P", "S", "T", "H", "L", "R", "U", "V"]
FEATURE_CLASS_OTHER = len(FEATURE_CLASSES)


class TrainData(ProductionData):
    def __init__(self, es_data, max_choices=25, max_codes=50,
                 oov_bucket_fix=False, feature_blocks=None,
                 full_null_row=False):
        super().__init__(es_data, max_choices, max_codes, oov_bucket_fix,
                         feature_blocks, full_null_row)
        self.labels, self.countries = self.create_labels(es_data)
        self.feature_classes = self.create_class_labels(es_data)

    def create_class_labels(self, es_data):
        """Feature class of the gold candidate, for the auxiliary head."""
        out = []
        for ex in es_data:
            idx = FEATURE_CLASS_OTHER
            if np.sum(ex['correct']) > 0:
                pos = np.where(np.array(ex['correct']))[0][0]
                fc = ex['es_choices'][pos].get('feature_class')
                if fc in FEATURE_CLASSES:
                    idx = FEATURE_CLASSES.index(fc)
            out.append(idx)
        return np.array(out).astype(np.int32)

    def __getitem__(self, index):
        return (self.labels[index],
                self.countries[index],
               {"placename_tensor": self.placename_tensor[index],  
                "doc_tensor": self.doc_tensor[index], 
                "other_locs_tensor": self.other_locs_tensor[index],
                "feature_codes": self.feature_codes[index], 
                "country_codes": self.country_codes[index],
                "gaz_info": self.gaz_info[index],
                "mask": self.mask[index],
                "class_label": self.feature_classes[index]}) 

    def create_labels(self, es_data):
        """Create an array with the location of the correct geonames entry"""
        all_labels = []
        all_countries = []
        for n, ex in enumerate(es_data):
            labels = np.zeros(self.max_choices)
            if np.sum(ex['correct']) == 0:
               labels[-1] = 1
               all_countries.append(self.country_dict["NULL"])
               # make an array of 0s of length 9 for gaz_info
            else:
                correct_num = np.where(np.array(ex['correct']))[0]
                if (self.full_null_row
                        and len(ex['es_choices']) > self.max_choices
                        and correct_num[0] >= self.max_choices - 1):
                    # The gold fell outside the window this example now shows
                    # (or is the gazetteer's own NULL row, which moved to the
                    # reserved slot): "no correct answer" is the right label.
                    correct_num = np.array([self.max_choices - 1])
                labels[correct_num] = 1
                try:
                    cn = correct_num[0]
                    country_code = ex['es_choices'][cn]['country_code3']
                    all_countries.append(self.country_dict[country_code])

                except Exception as e:
                    print(e)
                    print("subsetting number: ", cn)
            ## HACK here: convert back to index, not one-hot
            labels = np.argmax(labels)
            all_labels.append(labels)
        all_labels = np.array(all_labels).astype(np.int32)
        all_countries = np.array(all_countries).astype(np.int32)
        return all_labels, all_countries



class geoparse_model(nn.Module):
    def __init__(self, device, 
                bert_size, 
                num_feature_codes, 
                country_size=24, 
                code_size=8, 
                dropout=0.2,
                mix_dim=24,
                country_pred=False,
                n_extra_features=0,
                return_logits=False,
                mask_padding=False,
                modern_mlp=False,
                mix_depth=2,
                residual=False,
                listwise=False,
                listwise_heads=4,
                aux_country=False,
                aux_class=False):
        super(geoparse_model, self).__init__()
        self.device = device
        self.country_pred = country_pred
        # All three default to the original behavior.
        # return_logits: skip the final softmax so CrossEntropyLoss gets logits
        #   instead of probabilities it log-softmaxes a second time.
        # mask_padding: knock padded candidate rows out of the softmax/loss.
        # modern_mlp: GELU mix MLP, and dropout only inside that MLP.
        self.return_logits = return_logits
        self.mask_padding = mask_padding
        self.modern_mlp = modern_mlp
        self.residual = residual
        # embeddings setup
        try:
            pt = os.path.dirname(os.path.realpath(__file__))
            fn = os.path.join(pt, "assets", "country_bert_768.npy")
        except NameError:
            fn = os.path.join("assets", "country_bert_768.npy")
        pretrained_country = np.load(fn)
        pretrained_country = torch.FloatTensor(pretrained_country)
        logger.debug("Pretrained country embedding dim: {}".format(pretrained_country.shape))
        self.code_emb = nn.Embedding(num_feature_codes, code_size)
        self.country_emb = nn.Embedding.from_pretrained(pretrained_country, freeze=True)
        self.country_embed_transform = nn.Linear(bert_size, country_size) 

        # text layers
        self.text_to_country = nn.Linear(bert_size, country_size) 
        self.context_to_country = nn.Linear(bert_size, country_size) 
        self.text_to_code = nn.Linear(bert_size, code_size) 

        # transformation layers
        # 4 cosine similarities + 9 original gazetteer features, plus one
        # column per enabled enrichment feature.
        gaz_feature_count = 13 + n_extra_features
        self.mix_linear = nn.Linear(gaz_feature_count, mix_dim) # number of comparisons --> mix 
        self.mix_linear2 = nn.Linear(mix_dim, mix_dim) # mix --> mix
        self.last_linear = nn.Linear(mix_dim, 1) # mix --> final
        self.mix_country = nn.Linear(pretrained_country.shape[0], pretrained_country.shape[0],
                                    bias=False)
        self.country_predict = nn.Linear(country_size, pretrained_country.shape[0],
                                    bias=False)
        
        # Anything below is built only when asked for, so the default model
        # draws exactly the same random numbers it always did.
        self.mix_extra = nn.ModuleList([nn.Linear(mix_dim, mix_dim)
                                        for _ in range(max(0, mix_depth - 2))])
        # One mask-aware attention layer over the candidate dimension: every
        # candidate sees the others before it is scored, so two entries for the
        # same name can push each other down instead of being scored blind.
        self.listwise_attn = (nn.MultiheadAttention(mix_dim, listwise_heads,
                                                    batch_first=True)
                              if listwise else None)
        # Auxiliary heads hang off the *text* projections, so the pressure lands
        # on the encoder side rather than on the ranking MLP.
        self.aux_country_head = (nn.Linear(country_size, pretrained_country.shape[0])
                                 if aux_country else None)
        self.aux_class_head = (nn.Linear(code_size, FEATURE_CLASS_OTHER + 1)
                               if aux_class else None)

        # activations and similarities
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout) 
        self.similarity = nn.CosineSimilarity(dim=2)
        #self.similarity_country = nn.CosineSimilarity(dim=2)

    def forward(self, input, return_aux=False):
        ## TODO: this whole forward pass can probably be done with fewer permutations and transposes.

        # Unpack the dictionary here. Sending the data to device within the forward
        # function isn't standard, but it makes the training loop code easier to follow.
        placename_tensor = input['placename_tensor']
        other_locs_tensor = input['other_locs_tensor']
        doc_tensor = input['doc_tensor']
        feature_codes = input['feature_codes']
        country_codes = input['country_codes']
        gaz_info = input['gaz_info']
        logger.debug("feature_code input shape:{}".format(feature_codes.shape))

        ###### Text info setup  ######
        ### Apply linear layers to each of the inputs (placename tensor, other locs tensor,
        ###  full document tensor)
        if self.modern_mlp:
            # Dropout on both sides of a cosine similarity mostly adds noise to
            # an angle; keep it in the mix MLP only.
            x = self.text_to_country(placename_tensor)
            x_code = self.text_to_code(placename_tensor)
            x_other_locs = self.context_to_country(other_locs_tensor)
            x_doc = self.context_to_country(doc_tensor)
        else:
            x = self.dropout(self.text_to_country(placename_tensor))
            x_code = self.dropout(self.text_to_code(placename_tensor))
            x_other_locs = self.dropout(self.context_to_country(other_locs_tensor))
            x_doc = self.dropout(self.context_to_country(doc_tensor))
        logger.debug(f"x shape: {x.shape}")

        ####### Gazetteer entries setup ######
        ### Set up all the comparisions
        if self.modern_mlp:
            fc = self.code_emb(feature_codes)
            # country_emb is frozen: dropping units of it is dropout on a
            # constant lookup table, not regularization of a learned layer.
            cc = self.country_embed_transform(self.country_emb(country_codes))
        else:
            fc = self.dropout(self.code_emb(feature_codes))
            cc = self.country_embed_transform(self.dropout(self.country_emb(country_codes)))
        # to match the stacked value below, rearrange so it's
        # (choices, batch_size, embed_size)
        fc = fc.permute(1, 0, 2)
        cc = cc.permute(1, 0, 2)
        logger.debug("cc shape: {}, fc shape: {}".format(cc.shape, fc.shape))

        # Next, turn x from (batch_size, choices) into (1, batch_size, choices)
        # so it can be broadcast into a similarity comparison with all the ys.
        x_stack_country = torch.unsqueeze(x, 0) 
        x_stack_code = torch.unsqueeze(x_code, 0) 
        x_stack_locs = torch.unsqueeze(x_other_locs, 0)
        x_stack_doc = torch.unsqueeze(x_doc, 0)
        logger.debug("x_stack_country shape: {}".format(x_stack_country.shape))
        # x_stack is (choices, batch_size, embed_size)
        
        ## Do the similiary comparisons
        cos_sim_country = self.similarity(x_stack_country, cc)
        cos_sim_code = self.similarity(x_stack_code, fc)
        cos_sim_other_locs = self.similarity(x_stack_locs, cc)
        cos_sim_doc = self.similarity(x_stack_doc, cc)
        logger.debug("cos_sim_country: {}, cos_sim_code: {}, cos_sim_doc: {}".format(cos_sim_country.shape, cos_sim_country.shape, cos_sim_doc.shape))
        # put all the similarities into the shape (batch size, choices)  
        cos_sim_country = torch.unsqueeze(torch.transpose(cos_sim_country, 0, 1), 2)
        cos_sim_code = torch.unsqueeze(torch.transpose(cos_sim_code, 0, 1), 2)
        cos_sim_other_locs = torch.unsqueeze(torch.transpose(cos_sim_other_locs, 0, 1), 2)
        cos_sim_doc = torch.unsqueeze(torch.transpose(cos_sim_doc, 0, 1), 2)
        logger.debug("cos_sim_country shape: {}".format(cos_sim_country.shape))
        both_sim = torch.cat((cos_sim_country, cos_sim_code, cos_sim_other_locs, cos_sim_doc, gaz_info), 2)
        # the gaz_info features are (batch_size, choices, 6), to make 10 in the last dim.
        logger.debug(f"concat shape: {both_sim.shape}")  # (batch_size, choices, 10)
        act = self.gelu if self.modern_mlp else self.sigmoid
        last = self.dropout(act(self.mix_linear(both_sim)))
        mixed = self.dropout(act(self.mix_linear2(last)))
        last = last + mixed if self.residual else mixed
        for layer in self.mix_extra:
            step = self.dropout(act(layer(last)))
            last = last + step if self.residual else step
        if self.listwise_attn is not None:
            # True marks a key to ignore. The reserved last row is always live,
            # so no query is ever left with nothing to attend to.
            kpm = (input['mask'] < 0.5) if 'mask' in input else None
            attended, _ = self.listwise_attn(last, last, last,
                                             key_padding_mask=kpm,
                                             need_weights=False)
            last = last + attended
        # after applying last_layer, the output is dim 1 per choice. Squeeze that to produce a 
        # final output that's (batch_size, choices).
        last = torch.squeeze(self.last_linear(last), dim=2)  
        logger.debug(f"last shape: {last.shape}")  # (batch_size, choices)
        if self.mask_padding:
            # Padded rows carry sentinel values (99s, or -1s in the last row)
            # and no candidate. Without this they still take probability mass,
            # and the loss spends gradient pushing them down.
            mask = input['mask']
            last = last.masked_fill(mask < 0.5, -1e9)
        # softmax over the choices dimension so each location's choices will sum to 1
        # ...unless the caller wants logits: CrossEntropyLoss log-softmaxes for
        # itself, so handing it probabilities softmaxes twice.
        out = last if self.return_logits else self.softmax(last)
        logger.debug("out shape: {}".format(out.shape))  # should be (batch_size, choices)  (44, 25) 
        # try the country prediction again...
        if return_aux:
            aux = {}
            if self.aux_country_head is not None:
                aux['country'] = self.aux_country_head(x)
            if self.aux_class_head is not None:
                aux['fclass'] = self.aux_class_head(x_code)
            return out, aux
        if self.country_pred:
            country_pred = self.country_predict(self.sigmoid(x))
            if not self.return_logits:
                country_pred = self.softmax(country_pred)
            return out, country_pred
        else:
            return out



