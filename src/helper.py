import logging
import os
import random
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import torch


def setup_logging(log_dir=None, log_level=logging.INFO):
    """
    Set up logging to console and optional file.
    """
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")
    else:
        logging.info("Logging to console only.")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)  # hash-based ops
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class ResultsDB:
    def __init__(self, db_file: str = "experiments.duckdb"):
        db_path = Path(db_file)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(db_path))
        self.con.execute(
            """
        CREATE TABLE IF NOT EXISTS runs (
          ts         TIMESTAMP DEFAULT now(),
          model      TEXT NOT NULL,
          dataset    TEXT NOT NULL,
          method     TEXT NOT NULL,
          n_layers   INTEGER NOT NULL,
          seqlen     INTEGER NOT NULL,
          seed       INTEGER NOT NULL,
          accuracy   DOUBLE NOT NULL,
          n_samples  BIGINT  NOT NULL,
          batchsize  INTEGER,
          nbsamples  INTEGER
        );
        """
        )
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_triple ON runs(model, dataset, method);")
        self.con.execute("CREATE INDEX IF NOT EXISTS idx_seed   ON runs(seed);")

    def log(
        self,
        *,
        model,
        dataset,
        method,
        n_layers,
        seqlen,
        seed,
        accuracy,
        n_samples,
        nbsamples,
        batchsize=None,
    ):
        # Upsert on (model, dataset, method, seed)
        self.con.execute(
            """
            DELETE FROM runs
            WHERE model=? AND dataset=? AND method=? AND seed=? AND n_layers=? AND seqlen=? AND nbsamples=?;
        """,
            [model, dataset, method, int(seed), int(n_layers), int(seqlen), int(nbsamples)],
        )
        self.con.execute(
            """
            INSERT INTO runs (model, dataset, method, seed, n_layers, seqlen, accuracy, n_samples, batchsize, nbsamples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
            [
                model,
                dataset,
                method,
                int(seed),
                int(n_layers),
                int(seqlen),
                float(accuracy),
                int(n_samples),
                None if batchsize is None else int(batchsize),
                None if nbsamples is None else int(nbsamples),
            ],
        )

    def export_parquet(self, path="runs.parquet"):
        self.con.execute(f"COPY (SELECT * FROM runs) TO '{path}' (FORMAT PARQUET);")

    def export_csv(self, path="runs.csv"):
        self.con.execute(
            f"COPY (SELECT * FROM runs) TO '{path}' (FORMAT CSV, HEADER, DELIMITER ',');"
        )
