

import sqlite3
import os
import random
import metricas
import pickle
import math 
#import tensorflow as tf
#from tensorflow import keras

# Ruta absoluta a la base de datos
DATABASE_FILE = os.path.join(os.path.dirname(__file__), "datos", "datos.db")


# MODEL_PATH = os.path.join(os.path.dirname(__file__), "two_tower_apps.keras")
# MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), "two_tower_mappings.pkl")

# # Caché en memoria para no cargar el modelo todo el tiempo
# _TWO_TOWER_MODEL = None
# _TWO_TOWER_MAPPINGS = None
### --- RECOMENDADOR USADO --- ###
RECOMENDADOR_ACTIVO = "ensamble_pares_usercf"  # opciones: "top","top_n", "pares","user_cf","dos_torres","dos_torres_k","ensamble_pares_usercf"

# ============================
# Funciones auxiliares SQL
# ============================

def sql_execute(query, params=None):
    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()
    if params:
        cur.execute(query, params)
    else:
        cur.execute(query)
    con.commit()
    con.close()

def sql_select(query, params=None):
    con = sqlite3.connect(DATABASE_FILE)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    if params:
        res = cur.execute(query, params)
    else:
        res = cur.execute(query)
    ret = res.fetchall()
    con.close()
    return ret


#==============================comento porque python anywhere no lo acepta
# def _cargar_two_tower():
#     global _TWO_TOWER_MODEL, _TWO_TOWER_MAPPINGS

#     if _TWO_TOWER_MODEL is not None and _TWO_TOWER_MAPPINGS is not None:
#         return _TWO_TOWER_MODEL, _TWO_TOWER_MAPPINGS

#     if not os.path.exists(MODEL_PATH):
#         raise Exception("Modelo two-tower no encontrado. Ejecuta train_two_tower_apps.py primero.")

#     if not os.path.exists(MAPPINGS_PATH):
#         raise Exception("Mappings two-tower no encontrados. Ejecuta train_two_tower_apps.py primero.")

#     model = keras.models.load_model(MODEL_PATH, compile=False)
#     with open(MAPPINGS_PATH, "rb") as f:
#         mappings = pickle.load(f)

#     _TWO_TOWER_MODEL = model
#     _TWO_TOWER_MAPPINGS = mappings
#     return model, mappings

# ====== Fin helpers de Badges / Autoridad ======

# ============================
# Usuarios e interacciones
# ============================

def crear_usuario(username):
    query = "INSERT INTO perfiles(username) VALUES (?) ON CONFLICT DO NOTHING;"
    sql_execute(query, [username])

def insertar_interacciones(username, rating=None, app_upvote=None, app_review=None, description=None):
    slug = app_review or app_upvote
    if not slug:
        print(f"[WARN] insertar_interacciones: sin slug válido para {username}")
        return "sin_slug"

    exists = sql_select("""
        SELECT 1 FROM interacciones
        WHERE username = ?
          AND (app_upvote = ? OR app_review = ?)
        LIMIT 1;
    """, [username, slug, slug])

    if exists:
        sql_execute("""
            UPDATE interacciones
            SET rating = ?,
                description = COALESCE(?, description),
                app_review = COALESCE(app_review, ?)
            WHERE username = ?
              AND (app_upvote = ? OR app_review = ?);
        """, [rating, description, slug, username, slug, slug])
        print(f"[DEBUG] UPDATE rating={rating} para slug={slug}, user={username}")
        return "update"
    else:
        sql_execute("""
            INSERT INTO interacciones (username, rating, app_upvote, app_review, description)
            VALUES (?, ?, ?, ?, ?);
        """, [username, rating, app_upvote, app_review, description])
        print(f"[DEBUG] INSERT rating={rating} para slug={slug}, user={username}")
        return "insert"

def reset_usuario(username):
    sql_execute("DELETE FROM interacciones WHERE username = ?;", [username])

# ============================
# Consultas
# ============================

def obtener_producto(slug):
    res = sql_select("SELECT * FROM productos WHERE slug = ?;", [slug])
    return res[0] if res else None

def items_valorados(username):
    query = """
        SELECT DISTINCT COALESCE(app_review, app_upvote) AS slug
        FROM interacciones
        WHERE username = ?
          AND (
                (rating IS NOT NULL AND rating > 0)
             OR (app_upvote IS NOT NULL AND app_upvote <> '')
             OR (description IS NOT NULL AND TRIM(description) <> '')
          );
    """
    rows = sql_select(query, [username])
    return [i["slug"] for i in rows if i["slug"]]

def items_vistos(username):
    rows = sql_select("""
        SELECT DISTINCT COALESCE(app_review, app_upvote) AS slug
        FROM interacciones
        WHERE username = ?
          AND rating = 0
          AND (app_review IS NOT NULL OR app_upvote IS NOT NULL)
        ;
    """, [username])

    return [r["slug"] for r in rows if r["slug"]]

def items_desconocidos(username):
    rows = sql_select("""
        SELECT p.slug
        FROM productos p
        WHERE p.slug NOT IN (
            SELECT DISTINCT COALESCE(app_review, app_upvote)
            FROM interacciones
            WHERE username = ?
        )
    """, [username])
    return [r["slug"] for r in rows]

def datos_productos(slugs):
    if not slugs:
        print("[INFO] datos_productos: lista vacía, no se ejecuta query.")
        return []
    placeholders = ",".join(["?"] * len(slugs))
    query = f"SELECT DISTINCT * FROM productos WHERE slug IN ({placeholders})"
    productos = sql_select(query, slugs)
    print(f"[DEBUG] datos_productos: {len(productos)} productos recuperados.")
    return productos

def obtener_top_slugs_desde_top_apps(max_apps=200):
    """
    Devuelve una lista de slugs de las apps más populares,
    leyendo directamente de la tabla top_apps.
    """
    rows = sql_select("""
        SELECT slug
        FROM top_apps
        ORDER BY cant DESC
        LIMIT ?;
    """, [max_apps])
    return [r["slug"] for r in rows if r["slug"]]

def calcular_similitud_items(
    min_rating=6,
    min_coocurrencias=20,
    max_usuarios=2000,
    max_apps_top=200
):
    """
    Versión *light* de la tabla de similitudes de apps basada en co-ocurrencia.

    AHORA:
    - Usa SOLO las `max_apps_top` apps más populares (tabla top_apps).
    - Sigue limitando a los `max_usuarios` con más interacciones positivas.
    - Guarda sólo pares con al menos `min_coocurrencias` co-ocurrencias.
    - Interacción positiva = rating >= min_rating o app_upvote no vacío.

    Crea/llena:
        item_similitudes(
            slug_1 TEXT,
            slug_2 TEXT,
            similitud INTEGER,
            PRIMARY KEY (slug_1, slug_2)
        )
    """

    print(f"⚖️ Versión light de item_similitudes "
          f"(max_usuarios={max_usuarios}, "
          f"min_coocurrencias={min_coocurrencias}, "
          f"min_rating={min_rating}, "
          f"max_apps_top={max_apps_top})")

    # 0) Si ya tiene datos, no volvemos a calcular
    result = sql_select("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='item_similitudes';
    """)
    if result:
        count = sql_select("SELECT COUNT(*) AS cnt FROM item_similitudes;")
        if count[0]["cnt"] > 0:
            print("✅ item_similitudes ya existe con datos, omitiendo creación")
            return

    print("🔄 Creando tabla item_similitudes...")
    sql_execute("DROP TABLE IF EXISTS item_similitudes;")
    sql_execute("""
        CREATE TABLE item_similitudes (
            slug_1   TEXT,
            slug_2   TEXT,
            similitud INTEGER,
            PRIMARY KEY (slug_1, slug_2)
        );
    """)

    # # 0.1) Aseguramos índices mínimos en interacciones
    # sql_execute("""
    #     CREATE INDEX IF NOT EXISTS idx_interacciones_username
    #     ON interacciones(username);
    # """)
    # sql_execute("""
    #     CREATE INDEX IF NOT EXISTS idx_interacciones_app_review
    #     ON interacciones(app_review);
    # """)
    # sql_execute("""
    #     CREATE INDEX IF NOT EXISTS idx_interacciones_app_upvote
    #     ON interacciones(app_upvote);
    # """)

    # 1) Aseguramos que exista top_apps (por si no se llamó a init())
    top_table = sql_select("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='top_apps';
    """)
    if not top_table:
        print("ℹ️ top_apps no existe, creándola (como en init())...")
        sql_execute("DROP TABLE IF EXISTS top_apps;")
        sql_execute("""
            CREATE TABLE top_apps AS
            SELECT
                COALESCE(app_review, app_upvote) AS slug,
                COUNT(*) AS cant
            FROM interacciones
            WHERE (rating > 0)
               OR (app_upvote IS NOT NULL AND app_upvote <> '')
            GROUP BY 1;
        """)
        sql_execute("CREATE INDEX IF NOT EXISTS idx_top_apps ON top_apps(slug);")

    # 2) Tomamos sólo las apps más populares desde top_apps
    top_slugs = obtener_top_slugs_desde_top_apps(max_apps_top)
    if not top_slugs:
        print("⚠️ No hay slugs en top_apps; no se calculan similitudes.")
        return

    print(f"📌 Usando {len(top_slugs)} apps de top_apps para calcular pares.")

    placeholders_slugs = ",".join(["?"] * len(top_slugs))

    print("📊 Calculando similitudes (versión light, filtrado por top_apps)...")

    # 3) Cálculo con usuarios limitados + apps limitadas
    sql = f"""
        WITH usuarios_top AS (
            SELECT username
            FROM interacciones
            WHERE COALESCE(app_review, app_upvote) IS NOT NULL
              AND COALESCE(app_review, app_upvote) <> ''
              AND (
                    rating >= {min_rating}
                 OR (app_upvote IS NOT NULL AND app_upvote <> '')
              )
            GROUP BY username
            ORDER BY COUNT(*) DESC          -- usuarios con más actividad
            LIMIT {max_usuarios}
        ),
        interacciones_filtradas AS (
            SELECT *
            FROM interacciones
            WHERE username IN (SELECT username FROM usuarios_top)
              AND COALESCE(app_review, app_upvote) IS NOT NULL
              AND COALESCE(app_review, app_upvote) <> ''
              AND (
                    rating >= {min_rating}
                 OR (app_upvote IS NOT NULL AND app_upvote <> '')
              )
              AND COALESCE(app_review, app_upvote) IN ({placeholders_slugs})
        )
        INSERT INTO item_similitudes (slug_1, slug_2, similitud)
        SELECT
            COALESCE(i1.app_review, i1.app_upvote) AS slug_1,
            COALESCE(i2.app_review, i2.app_upvote) AS slug_2,
            COUNT(*) AS similitud
        FROM interacciones_filtradas i1
        JOIN interacciones_filtradas i2
          ON i1.username = i2.username
         AND COALESCE(i1.app_review, i1.app_upvote)
             < COALESCE(i2.app_review, i2.app_upvote)  -- evita (A,B) y (B,A)
        GROUP BY slug_1, slug_2
        HAVING COUNT(*) >= {min_coocurrencias}
        ORDER BY similitud DESC;
    """

    sql_execute(sql, top_slugs)

    count = sql_select("SELECT COUNT(*) AS cnt FROM item_similitudes;")
    print(f"✅ item_similitudes (light + top_apps) creada con {count[0]['cnt']} pares de similitudes")


def ensure_item_similitudes():
    """
    Se asegura de que la tabla item_similitudes exista **y tenga datos**.
    Si no existe o está vacía, llama a calcular_similitud_items().
    """
    result = sql_select("""
        SELECT name 
        FROM sqlite_master
        WHERE type='table' AND name='item_similitudes';
    """)

    if not result:
        print("⚠️ item_similitudes no existe, generándola...")
        calcular_similitud_items()  # usa los defaults, ahora con top_apps
        return

    # La tabla existe: miro cuántas filas tiene
    count = sql_select("SELECT COUNT(*) AS cnt FROM item_similitudes;")
    if count[0]["cnt"] == 0:
        print("⚠️ item_similitudes está vacía, recalculando (versión light)...")
        calcular_similitud_items()
    else:
        print(f"✅ item_similitudes OK: {count[0]['cnt']} pares de similitud")


# ======================================================
# Optimización: cargar item_similitudes SOLO una vez
# ======================================================

ITEM_SIM_READY = False

def init_item_similitudes():
    """
    Asegura que la tabla item_similitudes se verifique SOLO una vez
    por ciclo de servidor (para evitar lentitud en PythonAnywhere).
    """
    global ITEM_SIM_READY
    if not ITEM_SIM_READY:
        ensure_item_similitudes()
        ITEM_SIM_READY = True


def init():
    # --- TOP (popularidad) ---
    print("init: top_apps")
    sql_execute("DROP TABLE IF EXISTS top_apps;")
    sql_execute("""
        CREATE TABLE top_apps AS
        SELECT
            COALESCE(app_review, app_upvote) AS slug,
            COUNT(*) AS cant
        FROM interacciones
        WHERE (rating > 0)
           OR (app_upvote IS NOT NULL AND app_upvote <> '')
        GROUP BY 1;
    """)
    sql_execute("CREATE INDEX IF NOT EXISTS idx_top_apps ON top_apps(slug);")



# ============================
# Recomendaciones
# ============================

def recomendar_top(username, productos_relevantes, productos_desconocidos, N=9):
    """
    Devuelve las N apps más populares (rating + upvotes),
    seleccionadas aleatoriamente entre las más destacadas
    que el usuario aún no vio.
    """
    query = """
        SELECT 
            COALESCE(app_review, app_upvote) AS slug,
            COUNT(*) AS total_interacciones,
            SUM(CASE WHEN rating > 0 THEN rating ELSE 0 END) AS suma_rating,
            COUNT(CASE WHEN app_upvote IS NOT NULL AND app_upvote <> '' THEN 1 END) AS total_upvotes
        FROM interacciones
        WHERE COALESCE(app_review, app_upvote) IS NOT NULL
        GROUP BY slug
        ORDER BY (suma_rating * 0.7 + total_upvotes * 0.3) DESC
        LIMIT 300;
    """
    populares = sql_select(query)
    desconocidos_set = set(productos_desconocidos)
    slugs_populares = [p["slug"] for p in populares if p["slug"] in desconocidos_set]

    # Elegimos N aleatorias del top popular (rotación)
    random.shuffle(slugs_populares)
    recomendados = slugs_populares[:N]
    print(f"[DEBUG] recomendar_top: {len(recomendados)} sugerencias generadas.")
    return recomendados

def recomendar_contexto(username, slug, productos_relevantes=None, productos_desconocidos=None, N=3):

    # Obtener relevancias si no vienen
    if productos_relevantes is None:
        productos_relevantes = items_valorados(username)
    if productos_desconocidos is None:
        productos_desconocidos = items_desconocidos(username)

    cant = len(productos_relevantes)

    # ----------------------------------------------
    # CASO 1 — Usuario sin historial
    # ----------------------------------------------
    if cant == 0:
        return recomendador_top_n_apps(
            username,
            productos_relevantes,
            productos_desconocidos,
            N
        )

    # ----------------------------------------------
    # CASO 2 — Usuario con poco historial (1–4 apps)
    # ----------------------------------------------
    if 1 <= cant <= 4:
        return recomendador_top_n_apps(
            username,
            productos_relevantes,
            productos_desconocidos,
            N
        )

    # ----------------------------------------------
    # CASO 3 — Usuario con historial suficiente (≥5)
    # → usar item–item (pares)
    # ----------------------------------------------
    return recomendar_pares_apps(
        username,
        productos_relevantes,
        productos_desconocidos,
        N
    )

def recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N=9):
    """
    Top-N estilo profe pero con apps.
    Lee de top_apps y excluye semillas (productos_relevantes).
    Opcional: filtra por productos_desconocidos después (para no romper tu flujo).
    """
    params = []
    sql = "SELECT slug FROM top_apps"

    if productos_relevantes:
        ph_rel = ",".join(["?"] * len(productos_relevantes))
        sql += f" WHERE slug NOT IN ({ph_rel})"
        params += list(productos_relevantes)

    # Traemos un colchón para luego filtrar por desconocidos sin quedarnos cortos
    sql += " ORDER BY cant DESC LIMIT ?;"
    params.append(max(N * 5, 50))  # colchón simple

    rows = sql_select(sql, params)
    slugs = [r["slug"] for r in rows]

 
    if productos_desconocidos:
        desconocidos = set(productos_desconocidos)
        slugs = [s for s in slugs if s in desconocidos]

    return slugs[:N]


def recomendar_pares_apps(username, productos_relevantes, productos_desconocidos, N=9):
    """
    Recomendador item–item por co-ocurrencias de apps (pares).
    """

    # 🔹 Asegurar inicialización SOLO UNA VEZ
    init_item_similitudes()

    # Si no tenemos base o candidatos, caemos a top-N
    if not productos_relevantes or not productos_desconocidos:
        return recomendador_top_n_apps(
            username,
            list(productos_relevantes or []),
            list(productos_desconocidos or []),
            N
        )

    productos_relevantes   = list(productos_relevantes)
    productos_desconocidos = list(productos_desconocidos)

    ph_rel  = ",".join(["?"] * len(productos_relevantes))
    ph_cand = ",".join(["?"] * len(productos_desconocidos))

    rows = sql_select(f"""
        SELECT slug, SUM(similitud) AS score
        FROM (
            SELECT slug_2 AS slug, similitud
            FROM item_similitudes
            WHERE slug_1 IN ({ph_rel})
              AND slug_2 IN ({ph_cand})

            UNION ALL

            SELECT slug_1 AS slug, similitud
            FROM item_similitudes
            WHERE slug_2 IN ({ph_rel})
              AND slug_1 IN ({ph_cand})
        )
        GROUP BY slug
        ORDER BY score DESC
        LIMIT ?;
    """,
        productos_relevantes
        + productos_desconocidos
        + productos_relevantes
        + productos_desconocidos
        + [N]
    )

    if not rows:
        return recomendador_top_n_apps(
            username,
            productos_relevantes,
            productos_desconocidos,
            N
        )

    return [r["slug"] for r in rows]


#===================user-based collaborative filtering========================================
def vector_usuario(username, min_rating=6):#si valoró o no
    """
    Devuelve el set de apps positivas (rating >= min_rating o upvote).
    """
    rows = sql_select("""
        SELECT COALESCE(app_review, app_upvote) AS slug
        FROM interacciones
        WHERE username = ?
          AND (
                rating >= ?
             OR (app_upvote IS NOT NULL AND app_upvote <> '')
          );
    """, [username, min_rating])

    return set(r["slug"] for r in rows)
import math

def similitud_usuarios(u_apps, v_apps):
    if not u_apps or not v_apps:
        return 0.0
    inter = len(u_apps & v_apps)
    denom = math.sqrt(len(u_apps) * len(v_apps))
    return inter / denom if denom > 0 else 0.0

def vecinos_similares(username, k=20):
    u_apps = vector_usuario(username)
    if not u_apps:
        return []

    # Obtengo todos los usuarios candidatos
    rows = sql_select("""
        SELECT DISTINCT username 
        FROM interacciones
        WHERE username <> ?
    """, [username])

    vecinos = []
    for r in rows:
        otro = r["username"]
        v_apps = vector_usuario(otro)

        s = similitud_usuarios(u_apps, v_apps)
        if s > 0:
            vecinos.append((otro, s))

    vecinos.sort(key=lambda x: x[1], reverse=True)
    return vecinos[:k]
def recomendar_user_based(username, relevantes_training, desconocidos, N=20, k=20):
    # VECTOR DEL USUARIO → SOLO TRAIN
    u_apps = set(relevantes_training)
    if not u_apps:
        return []

    # vecinos
    rows = sql_select("""
        SELECT DISTINCT username
        FROM interacciones
        WHERE username <> ?
    """, [username])

    vecinos = []
    for r in rows:
        otro = r["username"]

        # apps positivas del vecino 
        v_apps = vector_usuario(otro)
        # similitud SOLO basada en train
        inter = len(u_apps & v_apps)
        denom = math.sqrt(len(u_apps) * len(v_apps)) if len(v_apps) else 0
        sim = inter / denom if denom else 0

        if sim > 0:
            vecinos.append((otro, sim))

    vecinos.sort(key=lambda x: x[1], reverse=True)
    vecinos = vecinos[:k]

    # candidatos
    candidatos = {}
    for vecino, sim in vecinos:
        v_apps = vector_usuario(vecino)

        for app in v_apps:
            if app in u_apps:
                continue
            if app not in desconocidos:
                continue
            candidatos[app] = candidatos.get(app, 0) + sim

    recomendados = sorted(candidatos.items(), key=lambda x: x[1], reverse=True)
    return [slug for slug, score in recomendados[:N]]

#================dos torres==================================
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

def construir_item_embeddings():
    print("📌 Construyendo embeddings de items (TF-IDF)...")

    rows = sql_select("""
    SELECT 
        slug,
        name AS title,
        (name || ' ' || tagline) AS description,
        '' AS tags
    FROM productos;
""")

    corpus = []
    slugs = []

    for r in rows:
        slugs.append(r["slug"])
        texto = f"{r['title']} {r['description']} {r['tags']}"
        corpus.append(texto)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    matriz = vectorizer.fit_transform(corpus)

    pickle.dump((vectorizer, slugs, matriz),
                open("item_embeddings.pkl", "wb"))

    print(f"✅ Embeddings creados para {len(slugs)} apps.")


def embedding_usuario_dos_torres(productos_relevantes):
    try:
        vectorizer, slugs, matriz = pickle.load(open("item_embeddings.pkl", "rb"))
    except:
        print("⚠️ No embeddings. Corre construir_item_embeddings().")
        return None

    if not productos_relevantes:
        return None

    indices = [slugs.index(s) for s in productos_relevantes if s in slugs]
    if not indices:
        return None

    emb = matriz[indices].mean(axis=0)
    return np.asarray(emb).reshape(-1)

from sklearn.metrics.pairwise import cosine_similarity

def recomendar_dos_torres(username, productos_relevantes, productos_desconocidos, N=20):
    # Load embeddings
    try:
        vectorizer, slugs, matriz = pickle.load(open("item_embeddings.pkl", "rb"))
    except:
        print("⚠️ No embeddings. Ejecuta construir_item_embeddings().")
        return []

    # User embedding basado SOLO en productos_relevantes (train)
    emb_user = embedding_usuario_dos_torres(productos_relevantes)
    if emb_user is None:
        return []

    # Filtrar candidatos
    candidatos = [s for s in productos_desconocidos if s in slugs]
    if not candidatos:
        return []

    idx_cands = [slugs.index(s) for s in candidatos]
    M = matriz[idx_cands]

    scores = cosine_similarity(emb_user.reshape(1, -1), M)[0]

    orden = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)

    return [slug for slug, score in orden[:N]]




#==========================================================================python anywhere no lo acepta
# def recomendar_dos_torres_k(username, productos_relevantes, productos_desconocidos, N=20):
#     """
#     Two-Tower con Keras, similar al de tu compañero.
#     - Usa embeddings aprendidos de usuario y app.
#     - Ranking sólo entre productos_desconocidos (como en el test).
#     """
#     try:
#         model, mappings = _cargar_two_tower()
#     except Exception as e:
#         print("⚠️ Two-tower no disponible:", e)
#         # fallback: por ejemplo user_cf o top_n
#         return recomendar_user_based(username, productos_relevantes, productos_desconocidos, N)

#     user_to_idx = mappings["user_to_idx"]
#     slug_to_idx = mappings["slug_to_idx"]

#     # Si el usuario no está en el modelo, fallback (similar a tu compañero)
#     if username not in user_to_idx:
#         print(f"⚠️ Usuario {username} no está en el modelo two-tower, usando fallback.")
#         return recomendar_user_based(username, productos_relevantes, productos_desconocidos, N)

#     # Candidatos: productos_desconocidos que existan en slug_to_idx
#     candidatos = [s for s in productos_desconocidos if s in slug_to_idx]
#     if not candidatos:
#         print("⚠️ No hay candidatos en el modelo two-tower, usando fallback.")
#         return recomendar_user_based(username, productos_relevantes, productos_desconocidos, N)

#     user_idx = user_to_idx[username]
#     user_array = np.array([user_idx] * len(candidatos), dtype="int32")
#     item_array = np.array([slug_to_idx[s] for s in candidatos], dtype="int32")

#     # Predicción de scores
#     scores = model.predict([user_array, item_array], verbose=0).flatten()

#     # Ordenar candidatos por score
#     orden = sorted(zip(candidatos, scores), key=lambda x: x[1], reverse=True)

#     # Devolver top-N slugs
#     return [slug for slug, score in orden[:N]]

#======ensamble entre pares de item y user
def recomendar_ensamble_pares_usercf(
    username,
    productos_relevantes,
    productos_desconocidos,
    N=20,
    alpha=0.5,           # peso de pares vs user_cf (0.5 = mitad y mitad)
    n_candidatos=100     # tamaño de la lista intermedia para combinar
):
    """
    Ensamble sencillo entre:
      - recomendar_pares_apps (item–item con item_similitudes)
      - recomendar_user_based (user-based CF)

    No recalcula item_similitudes, solo usa lo que ya tenés en la base.
    Combina las listas por ranking (no por score real) para no tocar el SQL.
    """

    # safety por si vienen None
    productos_relevantes = list(productos_relevantes or [])
    productos_desconocidos = list(productos_desconocidos or [])

    if not productos_relevantes or not productos_desconocidos:
        # si no hay info, caemos a top_n
        return recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N)

    # 1) Recomendaciones por pares (item-item)
    rec_pares = recomendar_pares_apps(
        username,
        productos_relevantes,
        productos_desconocidos,
        N=min(n_candidatos, len(productos_desconocidos))
    )

    # 2) Recomendaciones user-based
    rec_user = recomendar_user_based(
        username,
        productos_relevantes,
        set(productos_desconocidos),
        N=min(n_candidatos, len(productos_desconocidos))
    )

    if not rec_pares and not rec_user:
        # fallback duro
        return recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N)

    # 3) Combinar por ranking (cuanto más arriba, más puntaje)
    scores = {}

    # función ayuda: puntaje según posición (rank 0 = mejor)
    def score_from_rank(rank, length):
        return max(length - rank, 1)

    # pares
    Lp = len(rec_pares)
    for rank, slug in enumerate(rec_pares):
        s = alpha * score_from_rank(rank, Lp)
        scores[slug] = scores.get(slug, 0.0) + s

    # user_cf
    Lu = len(rec_user)
    for rank, slug in enumerate(rec_user):
        s = (1 - alpha) * score_from_rank(rank, Lu)
        scores[slug] = scores.get(slug, 0.0) + s

    # 4) Ordenar por score combinado
    ordenados = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 5) Devolver top-N que sigan siendo desconocidos (por las dudas)
    resultado = []
    desconocidos_set = set(productos_desconocidos)
    for slug, _ in ordenados:
        if slug in desconocidos_set:
            resultado.append(slug)
        if len(resultado) >= N:
            break

    # fallback si quedó vacío por alguna razón rara
    if not resultado:
        return recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N)

    return resultado




# def recomendar(username, productos_relevantes=None, productos_desconocidos=None, N=500):
#     if not productos_relevantes:
#         productos_relevantes = items_valorados(username)

#     if not productos_desconocidos:
#         productos_desconocidos = items_desconocidos(username)

    # if RECOMENDADOR_ACTIVO == "top":
    #     return recomendar_top(username, productos_relevantes, productos_desconocidos, N)
    # elif RECOMENDADOR_ACTIVO == "top_n":
    #      return recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N)
    # elif RECOMENDADOR_ACTIVO == "pares":
    #      return recomendar_pares_apps(username, productos_relevantes, productos_desconocidos, N)
    # elif RECOMENDADOR_ACTIVO == "user_cf":
    #     return recomendar_user_based(username, productos_relevantes, productos_desconocidos, N)
    # elif RECOMENDADOR_ACTIVO == "dos_torres":
    #     return recomendar_dos_torres(username, productos_relevantes, productos_desconocidos, N)
    # elif RECOMENDADOR_ACTIVO == "dos_torres_k":
    #     return recomendar_dos_torres_k(username, productos_relevantes, productos_desconocidos, N)
    # elif RECOMENDADOR_ACTIVO == "ensamble_pares_usercf":
    #     return recomendar_ensamble_pares_usercf(username, productos_relevantes, productos_desconocidos, N)
    # else:
    #     raise ValueError(f"Recomendador '{RECOMENDADOR_ACTIVO}' no reconocido")



# def recomendar(username, productos_relevantes=None, productos_desconocidos=None, N=500):
#     if not productos_relevantes:
#         productos_relevantes = items_valorados(username)

#     if not productos_desconocidos:
#         productos_desconocidos = items_desconocidos(username)

#     # ---------------------------
#     # 🔵 Lógica automática final
#     # ---------------------------
#     n_ratings = len(productos_relevantes)

#     if n_ratings == 0:
#         # Usuario completamente nuevo → recomendador seguro
#         return recomendar_top(username, productos_relevantes, productos_desconocidos, N)

#     if n_ratings < 5:
#         # Usuario con poca información → top_n
#         return recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N)

#     # Usuario con suficiente señal → ensamble definitivo
#     return recomendar_ensamble_pares_usercf(
#         username,
#         productos_relevantes,
#         productos_desconocidos,
#         N
#     )
def recomendar(username, productos_relevantes=None, productos_desconocidos=None, N=20):

    if not productos_relevantes:
        productos_relevantes = items_valorados(username)

    if not productos_desconocidos:
        productos_desconocidos = items_desconocidos(username)

    # 1) modelo principal
    slugs = recomendar_ensamble_pares_usercf(username, productos_relevantes, productos_desconocidos, N)

    # 2) fallback user_cf
    if not slugs:
        slugs = recomendar_user_based(username, productos_relevantes, productos_desconocidos, N)

    # 3) fallback popularidad
    if not slugs:
        slugs = recomendador_top_n_apps(username, productos_relevantes, productos_desconocidos, N)

    return slugs





# ============================
# Test / Métricas
# ============================

def test(username):
    apps_relevantes = items_valorados(username)
    apps_desconocidas = items_vistos(username) + items_desconocidos(username)

    random.shuffle(apps_relevantes)
    corte = int(len(apps_relevantes) * 0.8)
    apps_relevantes_training = apps_relevantes[:corte]
    apps_relevantes_testing  = apps_relevantes[corte:] + apps_desconocidas

    recomendacion = recomendar(username, apps_relevantes_training, apps_relevantes_testing, 20)

    relevance_scores = []
    for slug in recomendacion:
        res = sql_select("""
            SELECT rating, app_upvote
            FROM interacciones
            WHERE username = ?
              AND (app_upvote = ? OR app_review = ?)
            LIMIT 1;
        """, [username, slug, slug])

        if res:
            r = res[0]
            rating_val = r["rating"] if r["rating"] is not None else 0
            up = r["app_upvote"]
            rel = 1 if (rating_val >= 6 or (up is not None and up != "")) else 0
        else:
            rel = 0

        relevance_scores.append(rel)

    score = metricas.normalized_discounted_cumulative_gain(relevance_scores)
    return score



if __name__ == "__main__":
    ##init()-- no es necesario recalcular top
    ##construir_item_embeddings()
    users = sql_select("""
        SELECT username
        FROM perfiles
        WHERE (
            SELECT COUNT(*)
            FROM interacciones i
            WHERE i.username = perfiles.username
              AND COALESCE(app_review, app_upvote) IS NOT NULL
        ) >= 100
        LIMIT 50;
    """)
    users = [u["username"] for u in users]

    scores = []
    for user in users:
        score = test(user)   # o posicional: test(user, 20)
        scores.append(score)
        print(f"{user} >> {score:.6f}")

    ndcg_mean = (sum(scores) / len(scores)) if scores else 0.0
    print(f"\nNDCG: {ndcg_mean:.6f} --> {RECOMENDADOR_ACTIVO}")
