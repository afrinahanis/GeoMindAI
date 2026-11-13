# GeoMindAI

APGCE GeoHackathon 2026 — Challenge 1: Exploration Copilot.

GeoMindAI is a geoscience assistant that connects natural-language questions to
Python tools for raw seismic, well data and geoscientist-interpreted datasets.
It presents answers, supporting tool results and generated figures in a
Streamlit workspace.

## Setup

Use the Python environment validated for your HPC installation. The dependency
list is not a fully pinned environment lock. Streamlit 1.50 or later within
major version 1 is required by the interface dependency specification.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The model is accessed through an OpenAI-compatible API. Configure the model
server URL and model name in Settings or through environment variables:

```bash
export LLM_BASE_URL="http://172.28.34.1:8000/v1"
export LLM_MODEL="google/gemma-4-26B-A4B-it"
```

These defaults describe the hackathon environment. A different installation
needs its own reachable endpoint and available model. The application does
not install or host the language model.

## Connect data

In **Settings**, configure the raw, interpreted and cache folders, then save.
Use **Project data → ⋯** to scan or ingest the configured files. The first raw
ingestion can take time. Review its warnings before relying on the results.
Alternatively, configure the folders before running ingestion:

```bash
export DATA_DIR="/path/to/raw_data"
export INTERP_DIR="/path/to/interpreted_data"
export CACHE_DIR="cache"
python ingest_all.py
```

Supported sources include SEG-Y stacks/inversions, LAS logs, well information,
deviation surveys, checkshots, fault shapefiles/CSV, stratigraphic polygons and
horizon point files. Data availability determines which tools can run.

Interpreted folder examples:

```text
data_export/
  Fault (SHP)/
  Fault (CSV)/
  Polygon (SHP)/
  Horizon/
```

Use `examples/data_config.example.json` and `examples/slice_times.example.csv`
for mapping and time-slice configuration. Horizon point files are expected to
provide X, Y and time. Confirm coordinate systems and time units for your data.

## Workspace

- **Project data:** search datasets, control map visibility and inspect details.
- **Map:** explore the configured project layers and fit the view to the data.
- **Chat:** ask questions, review answers, generated figures and evidence, and
  select suggested follow-up questions when available.
- **Chat data menu:** choose datasets to focus a question. This supplies a focus
  hint to the assistant; it is not a strict tool-access or security restriction.
- **New chat:** start a separate conversation.
- **Chat history:** open the expanded chat view. Recent conversations are in the sidebar.
- **Settings:** configure paths, ingestion, the model endpoint and coordinates.

Suggested follow-ups are generated after new answers and may be omitted when
generation fails or no suitable question is returned. The extra model request
has a 12-second timeout. Older saved answers may not contain suggestions.

Example questions:

- What data is available and what evidence is missing?
- Compare the available well-log and seismic evidence for these wells.
- Does the evidence support a structural or stratigraphic trap, and what remains uncertain?

The default submission does not include the separately prepared reflector
tracker integration. Raw section/time-slice rendering remains available through
the existing tools. `horizon_slice` produces a thin time-window attribute map;
it does not follow a picked geological horizon.

## Files

| File or folder | Purpose |
|---|---|
| `app.py`, `geomind_ui/` | Streamlit interface and conversation persistence |
| `agent.py` | Model calls, tool orchestration and follow-up generation |
| `tools.py` | Registered tools and their parameter schemas |
| `ingest.py`, `interp_data.py`, `ingest_all.py` | Data ingestion, caching and refresh |
| `interpret.py`, `interp_tools.py`, `volmath.py` | Evidence calculations and screening |
| `ai_interp.py`, `models/fault_unet.pt` | Synthetic-training utility, fault model and facies tools |
| `run_copilot.ipynb` | Notebook workflow and tool examples |
| `app_classic.py` | Earlier alternative interface |
| `tests/` | Synthetic-data generators for development |

## Data flow and limitations

Source data is read from configured folders. Processed caches and figures are
written locally. Tool evidence, question/history text and some generated images
are sent to the configured language-model server. Whether this remains inside
the HPC or company network depends on that server's deployment and access rules.

Settings and chat history are stored in `.geomind_ui/` by default; use
`GEOMIND_HOME` to change that location. The ignore file excludes local data,
caches, histories, virtual environments and credentials from future staging.
It does not remove files that Git already tracks.

- Results support screening and require geoscientist review.
- Screening scores and relative confidence labels are not calibrated probabilities of success.
- Inadequate or conflicting evidence can lead to an insufficient-evidence conclusion.
- Wells without their own checkshots may have low-confidence time-depth ties.
- Confirm CRS, survey coverage, SEG-Y header mappings and time units.
- Automatically picked horizons and fault candidates need section-level review.
- Closure heights are in time units unless depth conversion is explicitly provided.
- Fault seal and missing geological information cannot be inferred from absent inputs.

## Validation and development disclosure

The repository includes synthetic-data generators and a bundled fault model.
Their presence alone is not a benchmark of accuracy on the field dataset.
Record the actual HPC environment, questions tested and observed limitations
when preparing the final submission. Capture its installed package versions
with `python -m pip freeze > requirements-hpc.txt` after validating the app.

AI coding assistance was used during development. Follow the event's required
format for declaring that assistance. Technical assumptions, source attribution
and limitations are retained in the code and documentation.

See [docs/AI_APPROACH.md](docs/AI_APPROACH.md) for the computational approach.
