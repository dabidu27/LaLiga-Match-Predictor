from dotenv import load_dotenv
import os
import psycopg2

load_dotenv()
url = os.getenv("database_url").replace('+psycopg2', '')

with psycopg2.connect(url) as conn:

    cursor = conn.cursor()
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS matches (
            date DATE,
            time VARCHAR(10),
            comp VARCHAR(50),
            round VARCHAR(50),
            day VARCHAR(15),
            venue VARCHAR(50),
            result VARCHAR(10),
            gf INT,
            ga INT,
            opponent VARCHAR(100),
            xg FLOAT,
            xga FLOAT,
            poss FLOAT,
            attendance INT,
            captain VARCHAR(50),
            formation VARCHAR(20),
            opp_formation VARCHAR(20),
            referee VARCHAR(50),
            match_report TEXT,
            notes TEXT,
            sh INT,
            sot INT,
            dist FLOAT,
            fk INT,
            pk INT,
            pkatt INT,
            name VARCHAR(100)
        );
                   """)
    cursor.execute("""
      CREATE TABLE IF NOT EXISTS upcoming_matches (
            date DATE,
            time VARCHAR(10),
            comp VARCHAR(50),
            round VARCHAR(50),
            day VARCHAR(15),
            venue VARCHAR(50),
            name VARCHAR(100),
            opponent VARCHAR(100),
            venue_code INT,
            hour INT,
            day_code INT);
        """)
            
    conn.commit()

print('Database created successfully')

