import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Wczytuje zmienne środowiskowe z pliku .env (gdzie są dane logowania)
load_dotenv()

# Pobiera konkretne parametry połączenia ze zmiennych środowiskowych
DB_SERVER = os.getenv('DB_SERVER')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# tworzenie połączenia z bazą danych
def get_db_engine():
    connection_string = (
        f"mssql+pymssql://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}"
    )
    return create_engine(connection_string)


# wczytuje dane z bazy oraz zmienia nazwy kolumn na bardziej czytelne
def load_train_data() -> pd.DataFrame:
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
    return pd.read_sql(query, engine)

if __name__ == "__main__":
    df_train = load_train_data()
    print(df_train.head())
