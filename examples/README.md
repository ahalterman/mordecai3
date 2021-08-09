## Production examples

This directory includes scripts that can be used to put Mordecai3 into production. Before using these files, you must install Mordecai3. From within this directory, you can run `pip install -e ../` to install Mordecai3 from local source.

### Batch Processing

The script `batch_process_production.py` can be used to batch process a JSON of stories with events, providing an event location for each event.

Usage Examples:

```
python batch_process_production.py storiesWithEvents.json config.ini
```

Where `storiesWithEvents.json` has entries that look like this:

```
{
  "Headline": "Bihar Cong leaders meet Rahul amid talks of change of guard",
  "Rawtext": "New Delhi, July 7 (IANS) Amid talks of change...",
  "events": [
    {
      "event_id": 35537115,
      "text": "New Delhi, July 7 (IANS) Amid talks of change ...",
      "name": "Consult",
      "sentence_num": 1
    },
    {
      "event_id": 35537116,
      "text": "The leaders then met...",
      "name": "Consult",
      "sentence_num": 2
    }
  ],
  "storyid": 51734911
}
```

It will return a CSV, with one row per event, with the following columns:

- event_id: from the input data
- storyid: from the input data
- text: full document text
- name: name of the event type
- sentence number: which sentence the event was coded from
- Headline: headline of the story
- extracted_name: the raw form of the place name that is identified as the event location
- mordecai_resolved_place: the canonical form of the event location
- mordecai_district: the district (ADM2) name of the event location
- mordecai_province: the province (ADM1) name of the event location
- mordecai_country: the country name of the event location
- mordecai_lat, mordecai_lon: lat and lon of the event location
- mordecai_geonameid: the Geonames ID for the event location
- mordecai_event_loc_reason: a short text description of how the location was selected
- tmp_partial_doc: [for debugging purposes] the full text up to and including the event sentence
- tmp_qa_answer: [for debugging purposes] the span of text identified by the QA model as the event location

The `config.ini` file includes full paths to the models used, along with tuning parameters for the geoparser.


### Interactive Use

For debugging and demo purposes, it can be useful to have an interactive version of Mordecai3. From within this directory, after installing Mordecai3 (`pip install -e ../`) and the Streamlit package, you can start up a browser-based dashboard of Mordecai3 like this:

```
streamlit run production_event.py --logger.level=INFO
```

The dashboard includes text geoparsing and will show which phrase was identified by the QA model as the event location. Not that this will differ somewhat from the production version, which uses a set of rules to tweak the event geolocation. 
