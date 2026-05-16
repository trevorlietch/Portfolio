-- Enable foreign key enforcement (must also be set per-connection in Python)
PRAGMA foreign_keys = ON;

-- frames: one row per dashcam image file
CREATE TABLE IF NOT EXISTS frames (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT    NOT NULL UNIQUE,          -- e.g. aspave_frame_0001.jpg
    relative_path TEXT   NOT NULL,                 -- path relative to pipeline/
    image_data   BLOB,                             -- raw image bytes
    image_mime_type TEXT,                          -- e.g. image/png
    image_size_bytes INTEGER,
    width        INTEGER,                          -- image width in pixels
    height       INTEGER,                          -- image height in pixels
    source       TEXT    DEFAULT 'aspave',         -- video source identifier
    frame_number INTEGER,                          -- numeric index from filename
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- annotations: driving labels for a frame
CREATE TABLE IF NOT EXISTS annotations (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id           INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    scene_description  TEXT,                       -- natural-language scene summary
    steering_angle_deg REAL    NOT NULL,           -- degrees: -180.0 to 180.0
    throttle           REAL    NOT NULL,           -- 0.0 (none) to 1.0 (full)
    brake              REAL    NOT NULL,           -- 0.0 (none) to 1.0 (full)
    annotation_source  TEXT    NOT NULL DEFAULT 'manual',  -- 'chatgpt' | 'manual'
    annotated_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at         TEXT    NOT NULL DEFAULT (datetime('now')),

    -- DB-level constraints mirror validate_turn_angle() from Driving Instructions Angle Test.py
    CONSTRAINT chk_steering CHECK (steering_angle_deg >= -180.0 AND steering_angle_deg <= 180.0),
    CONSTRAINT chk_throttle CHECK (throttle >= 0.0 AND throttle <= 1.0),
    CONSTRAINT chk_brake    CHECK (brake    >= 0.0 AND brake    <= 1.0)
);

-- label_categories: detected object types per annotation
CREATE TABLE IF NOT EXISTS label_categories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
    category      TEXT    NOT NULL,   -- one of the five standard categories below
    present       INTEGER NOT NULL DEFAULT 0,   -- 1 = detected, 0 = not detected
    confidence    REAL,                          -- optional model confidence (0.0-1.0)
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),

    CONSTRAINT chk_category CHECK (
        category IN ('pedestrian', 'vehicle', 'traffic_light', 'lane_marking', 'obstacle')
    ),
    UNIQUE (annotation_id, category)   -- one row per category per annotation
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_annotations_frame_id
    ON annotations(frame_id);

CREATE INDEX IF NOT EXISTS idx_label_categories_annotation_id
    ON label_categories(annotation_id);

CREATE INDEX IF NOT EXISTS idx_label_categories_category
    ON label_categories(category);

CREATE INDEX IF NOT EXISTS idx_frames_frame_number
    ON frames(frame_number);

CREATE INDEX IF NOT EXISTS idx_frames_source
    ON frames(source);

-- Alpamayo predicted paths
CREATE TABLE IF NOT EXISTS alpamayo_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL,
    model_name TEXT NOT NULL,
    nav_command TEXT NOT NULL,
    command_text TEXT,
    nav_command_source TEXT NOT NULL,
    selection_mode TEXT NOT NULL,
    selected_sample_index INTEGER NOT NULL,
    num_traj_samples INTEGER NOT NULL,
    guidance_weight REAL NOT NULL,
    max_generation_length INTEGER NOT NULL,
    frames_requested INTEGER NOT NULL,
    frames_stored INTEGER NOT NULL,
    reasoning_text TEXT,
    cot TEXT,
    selected_path_json TEXT NOT NULL,
    all_samples_json TEXT,
    gt_path_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS alpamayo_prediction_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES alpamayo_predictions(id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    UNIQUE(prediction_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_alpamayo_predictions_frame_id
    ON alpamayo_predictions(frame_id);

CREATE INDEX IF NOT EXISTS idx_alpamayo_prediction_points_prediction_id
    ON alpamayo_prediction_points(prediction_id);
