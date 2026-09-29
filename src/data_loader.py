import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_SERVER = os.getenv('DB_SERVER')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Dictionary mapping raw CSV columns to clean feature names
TRAIN_RENAME_MAP = {
    "battery_power": "battery_capacity_mah",
    "blue": "has_bluetooth",
    "clock_speed": "cpu_clock_speed_ghz",
    "dual_sim": "has_dual_sim",
    "fc": "front_camera_megapixels",
    "four_g": "has_4g",
    "int_memory": "internal_memory_gb",
    "m_dep": "phone_thickness_cm",
    "mobile_wt": "phone_weight_g",
    "n_cores": "cpu_cores_count",
    "pc": "primary_camera_megapixels",
    "px_height": "screen_pixel_height",
    "px_width": "screen_pixel_width",
    "ram": "ram_capacity_mb",
    "sc_h": "screen_height_cm",
    "sc_w": "screen_width_cm",
    "talk_time": "max_talk_time_hours",
    "three_g": "has_3g",
    "touch_screen": "has_touch_screen",
    "wifi": "has_wifi",
    "price_range": "price_category",
}

TEST_RENAME_MAP = {"id": "phone_id", **TRAIN_RENAME_MAP}

def get_db_engine():
    """Create and return a SQLAlchemy database engine."""
    connection_string = (
        f"mssql+pymssql://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}"
    )
    return create_engine(connection_string)


def load_train_data() -> pd.DataFrame:
    """Load raw training data from MS SQL Server or fallback to local CSV."""
    try:
        engine = get_db_engine()

        query = """
            SELECT 
                battery_power AS battery_capacity_mah,
                blue AS has_bluetooth,
                clock_speed AS cpu_clock_speed_ghz,
                dual_sim AS has_dual_sim,
                fc AS front_camera_megapixels,
                four_g AS has_4g,
                int_memory AS internal_memory_gb,
                m_dep AS phone_thickness_cm,
                mobile_wt AS phone_weight_g,
                n_cores AS cpu_cores_count,
                pc AS primary_camera_megapixels,
                px_height AS screen_pixel_height,
                px_width AS screen_pixel_width,
                ram AS ram_capacity_mb,
                sc_h AS screen_height_cm,
                sc_w AS screen_width_cm,
                talk_time AS max_talk_time_hours,
                three_g AS has_3g,
                touch_screen AS has_touch_screen,
                wifi AS has_wifi,
                price_range AS price_category
            FROM dbo.mobile_train
            """
        df = pd.read_sql(query, engine)
        print("Train data successfully loaded from database.")
        return df
    except Exception as e:
        print(f"Database connection failed ({e}). Loading fallback train.csv...")
        train_path = os.path.join("data", "train.csv")
        df = pd.read_csv(train_path)
        return df.rename(columns=TRAIN_RENAME_MAP)


def load_test_data() -> pd.DataFrame:
    """Load raw test data from MS SQL Server or fallback to local CSV."""
    try:
        engine = get_db_engine()

        query = """
            SELECT 
                id AS phone_id,
                battery_power AS battery_capacity_mah,
                blue AS has_bluetooth,
                clock_speed AS cpu_clock_speed_ghz,
                dual_sim AS has_dual_sim,
                fc AS front_camera_megapixels,
                four_g AS has_4g,
                int_memory AS internal_memory_gb,
                m_dep AS phone_thickness_cm,
                mobile_wt AS phone_weight_g,
                n_cores AS cpu_cores_count,
                pc AS primary_camera_megapixels,
                px_height AS screen_pixel_height,
                px_width AS screen_pixel_width,
                ram AS ram_capacity_mb,
                sc_h AS screen_height_cm,
                sc_w AS screen_width_cm,
                talk_time AS max_talk_time_hours,
                three_g AS has_3g,
                touch_screen AS has_touch_screen,
                wifi AS has_wifi
            FROM dbo.mobile_test
            """
        df = pd.read_sql(query, engine)
        print("Test data successfully loaded from database.")
        return df
    except Exception as e:
        print(f"Database connection failed ({e}). Loading fallback test.csv...")
        test_path = os.path.join("data", "test.csv")
        df = pd.read_csv(test_path)
        return df.rename(columns=TEST_RENAME_MAP)


if __name__ == "__main__":
    df_train = load_train_data()
    print(df_train.head())
    df_test = load_test_data()
    print(df_test.head())