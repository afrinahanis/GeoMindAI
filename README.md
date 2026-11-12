# Exploration Copilot (APGCE Geohackathon 2026, Challenge 1)

An agentic geoscience copilot. A geoscientist asks questions in plain language; the copilot (Gemma 4) calls
evidence tools on **raw data** (SEG-Y stacks and inversions, LAS logs, deviations, checkshots) and on
**interpreted data** (fault shapefiles, stratigraphic polygons, horizon), cross-checks them, and explains its reasoning step by step.

## Quick start
```
pip install -r requirements.txt
export DATA_DIR="/data/epic20/APGCE/input_data/challenge1_challenge2"                       # raw
export INTERP_DIR="/data/epic20/APGCE/apgce_team_11/insight-geohackathon/data_export"      # interpreted
python ingest_all.py            # first full ingestion (either folder may be missing)
streamlit run app.py            # or open run_copilot.ipynb
```
Data stays where it is; nothing is copied. Caches go to `CACHE_DIR` (default `cache/`).

## Interpreted data layout (auto-detected, names are flexible)
```
data_export/
  Fault (SHP)/timeslice_fault_01/*.shp ...      -> fault traces per time slice
  Polygon (SHP)/timeslice_channel_01/*.shp ...  -> sand-prone bodies
  Polygon (SHP)/timeslice_overbank_01/*.shp ... -> fine-grained / seal bodies
  Fault (CSV)/IL_timeslice_Fault_01.csv, CL_timeslice_Fault_01.csv ... + TWT_timeslice_Fault.csv
                                                 -> fault points as inline/xline per slice + slice times (layout auto-detected)
  Horizon/*.xyt                                  -> x y time points
```
Any new `timeslice_<class>_NN` folder is picked up as a new class. Slice times: put `slice_times.csv` (name,time_ms) or `data_config.json`
in `INTERP_DIR` (see `examples/`), or the time is read from an attribute, shapefile Z, or file name. Without times the tools still run but
time-dependent checks are skipped and the warning says so.

## Dynamic data
Add files to either folder, then press **Scan for new data** (app), run `ingest_all.refresh()` (notebook), or just ask the copilot
"new data was added": it calls `scan_for_new_data`. Only new or changed files (size + modification time) are re-read.
Raw only, interpreted only, or both all work.

## Tools (41)
Raw: list_wells, get_well_summary, get_log_stats, get_volume_info, sample_volumes_at_well, crosscheck_well, compare_wells, rank_wells,
stack_amplitudes_at_well, lateral_variability, render_slice, search_knowledge, logs_to_time, find_gas_zones, thickest_gas_zone,
compare_zone_to_seismic, extrapolate_zone_to_map, find_prospects, structure_and_closure, depositional_pattern, trap_evidence,
horizon_contour_and_faults, horizon_slice, overlay_structure_and_slice, high_variation_slices, count_faults, ai_fault_detection, ai_facies_map.

Interpreted / combined: list_datasets, scan_for_new_data, interpreted_faults, interpreted_facies, horizon_structure, interpreted_map,
well_context, validate_interpreted_vs_seismic, assess_traps, rank_prospects, recommend_trap (committed answer + reasoning chain), volume_math (arithmetic on volumes in a window, e.g. far - near), gas_potential (far-near negative + structural high + high sand probability).

## Files
| File | Role |
|---|---|
| `fault_csv.py` | reads Fault (CSV) IL/CL/TWT tables |
| `ingest.py` / `interp_data.py` / `ingest_all.py` | raw ingestion / interpreted ingestion / one entry point + incremental refresh |
| `interpret.py` | classical evidence computation on raw data |
| `ai_interp.py` | PyTorch 3D U-Net fault detection, k-means facies |
| `interp_tools.py` | reasoning over interpreted data + cross-validation with raw |
| `tools.py` | tool registry and JSON schemas given to Gemma |
| `agent.py` | Gemma tool-calling loop and system prompt |
| `app.py`, `run_copilot.ipynb` | chat UI and notebook |
| `docs/AI_APPROACH.md` | which AI is used, why, and the business value |
| `tests/make_synthetic_interpreted.py` | builds demo interpreted data to test without the real files |

## Honest limits
- Tested on synthetic shapefiles/horizon built in the expected layout; real attribute names, CRS and slice-time encoding may need `data_config.json`.
- Trap scores are screening scores from the evidence shown, not probabilities of success.
- Fault seal, depth conversion, unconformity and reef assessment need data that is not in the inputs and are reported as NOT AVAILABLE.
- Two-way time only; closure heights are in ms unless depth conversion is added.

- Automatic fault detection masks the survey-edge band and fills empty footprint cells; interpreted faults (shapefile or CSV) take priority for fault counts.
- Check how a Fault (CSV) folder was read: `python fault_csv.py "<INTERP_DIR>/Fault (CSV)"`.


## GeoMindAI web application (UI)
`streamlit run app.py` opens GeoMindAI: **Workspace** (available data, map, overview, chat button), **Chat history** (full-page chat) and **Settings** (raw / interpreted / cache folders, ingestion behaviour, model server, coordinates).
The UI lives in `geomind_ui/` and only calls the existing modules (`agent.run`, `ingest_all`, `interp_data`, `tools`); no analysis logic was changed. The previous single-page UI is kept as `app_classic.py`.
Extra dependency: `plotly` (map). Settings and chat history are stored in `.geomind_ui/` (override with env `GEOMIND_HOME`).
Ingestion after upload: interpreted files are ingested automatically (incremental, light); raw volumes are ingested on confirmation (heavy). Both are switchable in Settings.
