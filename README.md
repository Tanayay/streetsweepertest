# ParkLink Street Sweeper Test

A simple web app that accepts a street photograph, runs a YOLO vehicle detector, marks detected vehicles, and returns the total number of visible cars, trucks, buses, and motorcycles.

## Run locally

```bash
python -m venv .venv
```

Activate the environment, then install and run:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

The YOLO model downloads automatically on the first detection.

## Deploy on Render

1. Sign into Render and select **New + → Blueprint**.
2. Connect the GitHub repository `Tanayay/streetsweepertest`.
3. Select the included `render.yaml` configuration.
4. Create the service and wait for the build to finish.
5. Open the generated Render URL and upload a street image.

## Current behavior

The first prototype counts all visible vehicles recognized as cars, trucks, buses, or motorcycles. It does not yet distinguish parked vehicles from vehicles moving in a traffic lane.

## Planned accuracy upgrade

Add an adjustable curbside polygon so only vehicles located inside the selected parking area are counted. For fixed street-sweeping cameras, save one polygon per camera location and reuse it for every image.
