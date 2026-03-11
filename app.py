import os
import time
import atexit
import logging
import re
import shutil
import subprocess
import cv2

from flask import Flask, render_template, Response, send_from_directory, jsonify, abort, request, send_file
from Detector import DetectorMovimento, DetectorScrcpy
from fnaf import aplicar_efeitos_fnaf

# ------------------------
# Configurações gerais
# ------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

os.makedirs("movimentos", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# Configuração das câmeras: webcam indices e "phone" para scrcpy
cameras = {
    "cam1": 0,
    "cam2": 1,
    "cam3": 2,
    "phone": "scrcpy"
}

# Detectores "lazy"
detectors = {}
scrcpy_detectors = {}

# ------------------------
# Funções de inicialização de detectores
# ------------------------
def get_detector(cam_id: str):
    """Retorna o detector (webcam) já inicializado."""
    if cam_id not in cameras:
        return None

    if cameras[cam_id] == "scrcpy":
        return get_scrcpy(cam_id)

    if cam_id not in detectors:
        idx = cameras[cam_id]
        detectors[cam_id] = DetectorMovimento(
            camera_index=idx,
            min_area=1500,
            salvar_intervalo=2,
            duracao_video=3
        )
        app.logger.info(f"Detector criado para {cam_id} -> index {idx}")
    return detectors[cam_id]


def get_scrcpy(device_id="phone"):
    """Inicializa e retorna um detector scrcpy."""
    if device_id not in scrcpy_detectors:
        det = DetectorScrcpy()
        det.start()
        scrcpy_detectors[device_id] = det
        app.logger.info(f"Detector scrcpy criado para {device_id}")
    return scrcpy_detectors[device_id]

# ------------------------
# Geração de frames MJPEG
# ------------------------
def gen_frames(cam_id: str):
    """Gera frames MJPEG para qualquer câmera (webcam ou scrcpy)."""
    detector = get_detector(cam_id)
    if detector is None:
        app.logger.error(f"Câmera inválida: {cam_id}")
        return

    while True:
        try:
            # Atualiza frames conforme o tipo
            if isinstance(detector, DetectorMovimento):
                detector.detectar()
                frame = detector.frame1
            else:  # DetectorScrcpy
                frame = detector.get_frame()

            if frame is None:
                time.sleep(0.05)
                continue

            # Aplica efeitos FNAF se houver
            try:
                frame = aplicar_efeitos_fnaf(frame)
            except Exception as e:
                app.logger.debug(f"Erro aplicar_efeitos_fnaf: {e}")

            # Converte para JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(0.02)
                continue

            # Cada parte do multipart contém headers simples; o Response também força no-cache
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except GeneratorExit:
            break
        except Exception as e:
            app.logger.exception(f"Erro gen_frames: {e}")
            time.sleep(0.1)
            continue

# ------------------------
# Auxiliares de vídeo
# ------------------------
def _transcode_to_mp4_if_needed(src_path: str) -> str:
    base, ext = os.path.splitext(src_path)
    mp4_path = base + "_web.mp4"  # evita substituir original

    # só cria mp4 se não existir
    if os.path.exists(mp4_path):
        return mp4_path

    if shutil.which("ffmpeg") is None:
        return src_path

    try:
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            mp4_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(mp4_path):
            return mp4_path
    except Exception as e:
        app.logger.warning(f"Transcoding failed for {src_path}: {e}")

    return src_path

def _send_file_partial(path: str):
    """Serve arquivos com suporte a Range requests."""
    if not os.path.exists(path):
        abort(404)

    file_size = os.path.getsize(path)
    range_header = request.headers.get("Range", None)

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
        else:
            start, end = 0, file_size - 1

        if start >= file_size:
            return Response(status=416)
        if end >= file_size:
            end = file_size - 1

        length = end - start + 1
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        rv = Response(data, 206, mimetype="video/mp4")
        rv.headers.add("Content-Range", f"bytes {start}-{end}/{file_size}")
        rv.headers.add("Accept-Ranges", "bytes")
        rv.headers.add("Content-Length", str(length))
        return rv
    else:
        return send_file(path, mimetype="video/mp4", as_attachment=False)

# ------------------------
# Rotas do Flask
# ------------------------
@app.route("/")
def index():
    return render_template("index.html", cameras=list(cameras.keys()))


@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    if cam_id not in cameras:
        abort(404)
    response = Response(
        gen_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    # Força browser não armazenar cache
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/api/cameras")
def api_cameras():
    return jsonify({"cameras": list(cameras.keys())})


@app.route("/galeria")
def galeria():
    imagens = sorted(os.listdir("movimentos"), reverse=True)
    imagens = [f for f in imagens if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    raw_videos = sorted(os.listdir("videos"), reverse=True)
    videos = []
    for v in raw_videos:
        ext = os.path.splitext(v)[1].lower()
        if ext in ('.mp4', '.avi', '.mov', '.webm', '.ogg'):
            caminho = os.path.join("videos", v)
            mp4_path = _transcode_to_mp4_if_needed(caminho)
            videos.append(os.path.basename(mp4_path))

    return render_template("galeria.html", imagens=imagens, videos=videos)


@app.route('/movimentos/<path:filename>')
def movimentos_files(filename):
    return send_from_directory('movimentos', filename)


@app.route("/videos/<path:filename>")
def videos_files(filename):
    requested = os.path.normpath(os.path.join("videos", filename))
    if not requested.startswith(os.path.abspath("videos")):
        abort(404)
    if not os.path.exists(requested):
        abort(404)

    ext = os.path.splitext(requested)[1].lower()
    served_path = _transcode_to_mp4_if_needed(requested) if ext != ".mp4" else requested
    return _send_file_partial(served_path)

# ------------------------
# Cleanup
# ------------------------
def cleanup():
    app.logger.info("Liberando detectores...")
    for key, det in list(detectors.items()):
        try:
            det.liberar()
            app.logger.info(f"Detector {key} liberado")
        except Exception as e:
            app.logger.debug(f"Erro liberar detector {key}: {e}")
        finally:
            detectors.pop(key, None)

    for key, det in list(scrcpy_detectors.items()):
        try:
            det.stop()
            app.logger.info(f"Detector Scrcpy {key} parado")
        except Exception as e:
            app.logger.debug(f"Erro parar scrcpy {key}: {e}")
        finally:
            scrcpy_detectors.pop(key, None)

atexit.register(cleanup)

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print("Servidor iniciado em: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)