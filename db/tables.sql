
CREATE TABLE IF NOT EXISTS tsdb_sma_energy(
    datetime timestamp,
    energy float,
    PRIMARY KEY(datetime)
);
SELECT create_hypertable('tsdb_sma_energy', 'datetime');

CREATE TABLE IF NOT EXISTS tsdb_sma_cleaned(
    id integer,
    datetime timestamp,
    energy_exported float,
    PRIMARY KEY(id, datetime)
);
SELECT create_hypertable('tsdb_sma_cleaned', 'datetime');

CREATE TABLE IF NOT EXISTS tsdb_cams_mera_cleaned(
    datetime timestamp,
    gh float,
    csky_ghi float,
    tdry float,
    consumption float,
    PRIMARY KEY(datetime)
);

SELECT create_hypertable('tsdb_cams_mera_cleaned', 'datetime');

CREATE TABLE IF NOT EXISTS tsdb_sma_weatherbit_cleaned(
    datetime timestamp,
    gh float,
    csky_ghi float,
    tdry float,
    energy float,
    PRIMARY KEY(datetime)
);

SELECT create_hypertable('tsdb_sma_weatherbit_cleaned', 'datetime');


CREATE TABLE IF NOT EXISTS tsdb_gfs_cleaned(
    id integer,
    datetime timestamp,
    datetime_accessed timestamp,
    DSWRF_surface float,
    TCDC_boundary float,
    TCDC_high float,
    TCDC_middle float,
    TCDC_low float,
    TCDC_total float,
    TCDC_convective float,
    TMP_surface float,
    PRIMARY KEY(id, datetime, datetime_accessed)
);

SELECT create_hypertable('tsdb_gfs_cleaned', 'datetime');

CREATE TABLE IF NOT EXISTS tsdb_prediction_consumption(
    id integer,
    datetime timestamp,
    value float,
    PRIMARY KEY(id, datetime)
);

SELECT create_hypertable('tsdb_prediction_consumption', 'datetime');

CREATE TABLE IF NOT EXISTS tsdb_prediction_production(
    id integer,
    datetime timestamp,
    value float,
    PRIMARY KEY(id, datetime)
);

SELECT create_hypertable('tsdb_prediction_production', 'datetime');