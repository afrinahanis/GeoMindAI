# GeoMindAI computational approach

## Model and tools

GeoMindAI uses a configured OpenAI-compatible model endpoint. The default model
identifier is `google/gemma-4-26B-A4B-it`. The identifier is a configuration value;
this repository does not establish the served model's architecture, licensing,
training history or performance. Confirm those details with the server operator
before making claims about them in the submission.

The model chooses registered Python tools, receives their results and writes
an explanation. Native tool calls are used when the endpoint accepts them;
a JSON-in-text protocol provides a fallback. Some generated figures are also
supplied as image input, which requires a compatible model/server.

Prompt instructions require numbers to come from tool results, identify missing
information and distinguish general knowledge. These instructions reduce risk;
they do not guarantee factual accuracy. Tool traces support review of answers.
The model is instructed to allow an insufficient-evidence conclusion. A tool's
screening recommendation is evidence to assess rather than a mandatory conclusion.

## Evidence processing

- **Raw data:** SEG-Y, well information, deviation surveys, checkshots and LAS logs.
- **Interpreted data:** supplied faults, stratigraphic polygons and horizon points.
- **Classical methods:** log statistics, time-depth conversion, volume sampling,
  attribute arithmetic, structure/closure screening and spatial comparisons.
- **Fault model:** a small 3D U-Net implementation with a synthetic seismic
  training utility. A checkpoint is bundled in `models/fault_unet.pt`.
- **Facies:** unsupervised k-means classes on available attributes. Clusters
  require geological interpretation and are not automatically formation labels.

The checkpoint's presence does not prove field accuracy or a particular training
step count. Training-run metadata and field-validation metrics are not established
by this documentation. The code reports its fault method; when its availability
check fails, it uses a coherence-based fallback. Other runtime failures still
require diagnosis and are not guaranteed to trigger that fallback.

Supplied interpretation is used where available and compared with raw evidence.
Disagreement and missing data should be reported. Automatically selected peak
times can switch between reflectors; they are not confirmed geological horizons.
`horizon_slice` is a thin time-window attribute map. The separately packaged
reflector tracker is not part of this submission cleanup.

## Follow-up suggestions

After an answer, a separate short model request can propose up to three
questions using the answer, available tool names and loaded-data summary.
The application filters duplicates, unsupported tool names and new numeric
identifiers absent from the question/answer. It cannot guarantee that every
suggestion is relevant or that all required data is available. Generation
failure omits suggestions without discarding the answer.

## Data handling

Raw files are read from configured locations. Caches, figures and chat histories
are written locally. Questions, conversation text, tool evidence and selected
figures are transmitted to the configured model endpoint. Network boundaries,
authentication and retention depend on the deployed server. Local ingestion
alone does not establish that all processing remains on the user's machine.

## Interpretation limits and validation

Screening scores are not probabilities of success. Time-depth ties without
well-specific checkshots carry uncertainty. Check CRS, survey coverage, time
units and header mappings before connecting well and seismic observations.
Fault seal and depth conversion need appropriate inputs and methods.

Synthetic-data tests check selected code paths. They do not demonstrate field
performance. Record HPC smoke checks, actual data coverage, model availability,
representative questions and limitations separately. Do not claim measured
accuracy, time savings or cost savings without supporting measurements.
