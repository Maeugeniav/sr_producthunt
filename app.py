from flask import Flask, request, render_template, make_response, redirect
import recomendar
from datetime import datetime


MODELO_ACTIVO = "Ensamble:Item-Item + User-Based CF"
app = Flask(__name__)
app.debug = True

@app.get('/')
def get_index():
    return render_template('login.html')

@app.post('/')
def post_index():
    username = request.form.get('username', None)
    if username:
        recomendar.crear_usuario(username)
        res = make_response(redirect("/recomendaciones"))
        res.set_cookie('username', username)
        return res
    return render_template('login.html')

# ==============================
# Página principal de recomendaciones
# ==============================
@app.get('/recomendaciones')
def get_recomendaciones():
    username = request.cookies.get('username')
    if not username:
        return redirect("/")

    productos_relevantes = recomendar.items_valorados(username)
    productos_desconocidos = recomendar.items_desconocidos(username)
    slugs = recomendar.recomendar(username, N=9)

    # 🟡 Si ya no quedan apps por ver
    if not slugs:
        mensaje = "🎉 Ya viste todas las apps disponibles."
        return render_template(
            "recomendaciones.html",
            productos_recomendados=[],
            username=username,
            cant_valorados=len(recomendar.items_valorados(username)),
            cant_vistos=len(recomendar.items_vistos(username)),
            mensaje=mensaje
        )

    # 🟢 Registrar como vistas solo si no hubo interacción
    for slug in slugs:
        existente = recomendar.sql_select("""
            SELECT 1
            FROM interacciones
            WHERE username = ?
              AND COALESCE(app_review, app_upvote) = ?
            LIMIT 1;
        """, [username, slug])

        if not existente:
            recomendar.insertar_interacciones(
                username=username,
                rating=0,
                app_review=slug
            )
            print(f"[DEBUG] Insertada vista nueva: {slug}")
        else:
            print(f"[DEBUG] Ya existía vista: {slug}")

    # 🟢 Obtener productos y métricas
    productos_recomendados = recomendar.datos_productos(slugs)
    cant_valorados = len(recomendar.items_valorados(username))
    cant_vistos = len(recomendar.items_vistos(username))

    return render_template(
        "recomendaciones.html",
        productos_recomendados=productos_recomendados,
        username=username,
        cant_valorados=cant_valorados,
        cant_vistos=cant_vistos
    )

# ==============================
# Recomendaciones basadas en un producto
# ==============================
@app.get('/recomendaciones/<string:slug>')
def get_recomendaciones_apps(slug):
    username = request.cookies.get('username')
    if not username:
        return redirect("/")

    slugs_relacionados = recomendar.recomendar_contexto(username, slug)

    for s in slugs_relacionados:
        existente = recomendar.sql_select("""
            SELECT 1
            FROM interacciones
            WHERE username = ?
              AND (
                  app_upvote = ?
                  OR app_review = ?
                  OR (rating IS NOT NULL AND rating <> 0)
              )
            LIMIT 1;
        """, [username, s, s])

        if not existente:
            recomendar.insertar_interacciones(
                username=username,
                rating=0,
                app_review=s
            )

    productos_recomendados = recomendar.datos_productos(slugs_relacionados)
    cant_valorados = len(recomendar.items_valorados(username))
    cant_vistos = len(recomendar.items_vistos(username))
    producto = recomendar.obtener_producto(slug)

    return render_template(
        "recomendaciones_apps.html",
        producto=producto,
        productos_recomendados=productos_recomendados,
        username=username,
        cant_valorados=cant_valorados,
        cant_vistos=cant_vistos
    )

# ==============================
# Procesar interacciones (ratings, upvotes, reviews)
# ==============================
@app.post('/recomendaciones')
def post_recomendaciones():
    username = request.cookies.get('username')
    if not username:
        return redirect("/")

    # Guardar ratings enviados
    for key in request.form.keys():
        if key.startswith("rating_"):
            slug = key.replace("rating_", "")
            rating = request.form.get(key)
            rating = int(rating) if rating and rating.isdigit() else None

            if rating and rating > 0:
                recomendar.insertar_interacciones(
                    username=username,
                    rating=rating,
                    app_review=slug
                )

    # 👉 Importante: NO recalculamos recomendaciones aquí
    # Solo volvemos a la página, donde GET /recomendaciones
    # marcará TODAS las apps como vistas y calculará nuevas.
    return make_response(redirect("/recomendaciones"))


# ==============================
# Reset de usuario
# ==============================
@app.get('/reset')
def get_reset():
    username = request.cookies.get('username')
    if not username:
        return redirect("/")
    recomendar.reset_usuario(username)
    return make_response(redirect("/recomendaciones"))

# ==============================
# Fecha de última actualización
# ==============================
fecha_actualizacion = datetime(2025, 11, 30).strftime("%d/%m/%Y")

@app.context_processor
def inject_globals():
    """Inyecta variables globales en todos los templates"""
    return dict(
        fecha_actualizacion=fecha_actualizacion,
        modelo_activo=MODELO_ACTIVO
    )



if __name__ == '__main__':
    app.run()
