import threading
import webbrowser
import time
from Detector import DetectorMovimento
from fnaf import aplicar_efeitos_fnaf, tocar_alerta
from app import app  # servidor Flask

import cv2
from alerta_gpio import disparar_alerta


# --- Cria detector central ---
detector = DetectorMovimento(min_area=1500, salvar_intervalo=2, camera_index=0)

# --- Thread para Flask ---
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True
flask_thread.start()

# --- Abrir navegador para o feed ---
time.sleep(1)  # espera Flask iniciar
webbrowser.open("http://127.0.0.1:5000/")

# --- Loop principal da câmera ---

try:
    while True:
        eventos = detector.detectar()
        frame = aplicar_efeitos_fnaf(detector.frame1)

        if eventos:
            disparar_alerta(0.5)  # LED ou buzzer
        time.sleep(0.1)


        # mostrar feed local (opcional)
        cv2.imshow("Feed CCTV - MiniFNAF", frame)

        for e in eventos:
            print(f"Movimento detectado às {e['tempo']}")
            print(f"Imagem: {e['imagem']}")
            print(f"Vídeo: {e['video']}")
            tocar_alerta()

        if cv2.waitKey(10) == 27:
            break

except KeyboardInterrupt:
    print("Saindo...")
    pass

finally:
    detector.liberar()