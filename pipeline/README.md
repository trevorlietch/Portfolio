## Database Management

An integrated SQLite database (`annotations.db`) is used for managing, filtering, and querying specifically extracted route dashcam frames prior to Alpamayo ingestion.

### Interacting with the CLI:
```bash
python annotate_route.py ../datasets/route_1/segment_00
python run_alpamayo.py ../datasets/route_1/segment_00
python create_alpamayo_video.py ../datasets/route_1/segment_00
python import_route_db.py ../datasets/route_1/segment_00 --overwrite
```

From the project root, run the same scripts as:

```bash
python pipeline/annotate_route.py datasets/route_1/segment_00
python pipeline/run_alpamayo.py datasets/route_1/segment_00
python pipeline/create_alpamayo_video.py datasets/route_1/segment_00
python pipeline/import_route_db.py datasets/route_1/segment_00 --overwrite
```

`run_alpamayo.py` writes per-frame prediction JSON into each segment's `predictions/`
folder. `create_alpamayo_video.py` builds the MP4 afterward from the route folder's
raw frames, annotations, ground truth paths, and prediction paths.

### Programmatic Python Access:
```python
from dataset_manager import DatasetManager

# Initialize the manager
db = DatasetManager()

# Example: Push a new telemetry-bound frame into records
db.add_frame("frame.jpg", "frames/frame.jpg")
```
