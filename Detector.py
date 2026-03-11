# Detector.py
import cv2
import datetime
import os
import csv
import time
import logging
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple, Optional

import subprocess
import cv2
import threading


class DetectorScrcpy:
    """Captura frames do scrcpy via pipe de vídeo."""

    def __init__(self, device_id: str = None):
        self.device_id = device_id
        self.cap = None
        self.frame = None
        self._running = False
        self.thread = None

    def start(self):
        if self._running:
            return
        cmd = ["scrcpy", "--stay-awake", "--no-control", "--max-size", "640", "--bit-rate", "2M", "--output", "-"]
        if self.device_id:
            cmd.insert(1, "-s")
            cmd.insert(2, self.device_id)

        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self._running:
            raw = self.proc.stdout.read(640 * 480 * 3)  # 640x480 RGB
            if not raw:
                time.sleep(0.05)
                continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((480, 640, 3))
            self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def get_frame(self):
        return self.frame

    def stop(self):
        self._running = False
        if self.proc:
            self.proc.terminate()
        self.proc = None

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DetectorMovimento")

# Garantir diretórios
for folder in ["movimentos", "videos"]:
    os.makedirs(folder, exist_ok=True)


class DetectorMovimento:
    def __init__(
            self,
            camera_index: int = 0,
            min_area: int = 2000,
            salvar_intervalo: int = 2,
            duracao_video: int = 3
    ):
        # CAP_DSHOW é ótimo para Windows
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.min_area = min_area
        self.salvar_intervalo = salvar_intervalo
        self.duracao_video = duracao_video
        self.ultimo_salvamento = datetime.datetime.min
        self.log_file = "movimentos/log.csv"

        self.frame1: Optional[np.ndarray] = None
        self.frame2: Optional[np.ndarray] = None

        self._inicializar_frames()
        self._preparar_csv()

    def _inicializar_frames(self) -> None:
        """Tenta capturar os dois primeiros frames."""
        for _ in range(5):
            ret1, f1 = self.cap.read()
            time.sleep(0.1)
            ret2, f2 = self.cap.read()
            if ret1 and ret2 and f1 is not None and f2 is not None:
                self.frame1, self.frame2 = f1, f2
                return
        logger.warning("Câmera não inicializou corretamente.")

    def _preparar_csv(self) -> None:
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w", newline="") as f:
                    csv.writer(f).writerow(["timestamp", "posicao", "imagem", "video"])
            except OSError:
                pass

    def _processar_contornos(self) -> List[np.ndarray]:
        """Calcula a diferença entre frames e retorna contornos."""
        # Verificação explícita de None para o PyCharm
        if self.frame1 is None or self.frame2 is None:
            return []

        f1_s = cv2.resize(self.frame1, (640, 480))
        f2_s = cv2.resize(self.frame2, (640, 480))

        diff = cv2.absdiff(f1_s, f2_s)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if cv2.contourArea(c) > self.min_area]

    def _gravar_video(self, caminho: str) -> None:
        """Grava clipe de vídeo. Ignora erro de referência do VideoWriter_fourcc."""
        try:
            ret, temp_f = self.cap.read()
            if not ret or temp_f is None:
                return

            h, w = temp_f.shape[:2]
            # Usamos o ignore aqui porque o PyCharm não lê bem os binários do cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            out = cv2.VideoWriter(caminho, fourcc, 10, (w, h))

            for _ in range(int(self.duracao_video * 10)):
                r, f = self.cap.read()
                if r and f is not None:
                    out.write(f)
            out.release()
        except (cv2.error, Exception):  # Catch genérico aqui apenas para garantir que o processo não pare
            logger.error("Falha ao gravar vídeo.")

    def _salvar_evento(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Salva mídia e gera log."""
        agora = datetime.datetime.now()
        ts = agora.strftime("%Y-%m-%d_%H-%M-%S")
        img_p, vid_p = f"movimentos/mov_{ts}.png", f"videos/mov_{ts}.mp4"

        try:
            # Convertendo para PIL para salvar imagem com retângulo
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            x, y, w, h = bbox
            ImageDraw.Draw(pil_img).rectangle([x, y, x + w, y + h], outline="red", width=3)
            pil_img.save(img_p)
        except (OSError, cv2.error):
            img_p = ""

        self._gravar_video(vid_p)

        try:
            with open(self.log_file, "a", newline="") as f:
                csv.writer(f).writerow([ts, f"{bbox}", img_p, vid_p])
        except OSError:
            pass

        return {"tempo": ts, "posicao": bbox, "imagem": img_p, "video": vid_p}

    def detectar(self) -> List[Dict[str, Any]]:
        """Executa um ciclo de detecção."""
        eventos: List[Dict[str, Any]] = []
        contornos = self._processar_contornos()

        agora = datetime.datetime.now()
        pode_salvar = (agora - self.ultimo_salvamento).total_seconds() > self.salvar_intervalo

        # Se frame2 for None, evitamos o acesso a .shape
        f2 = self.frame2
        if contornos and pode_salvar and f2 is not None:
            c = max(contornos, key=cv2.contourArea)
            # Garantimos que f2 é ndarray para o linter
            h_orig, w_orig = f2.shape[:2]
            x_r, y_r = w_orig / 640, h_orig / 480

            bx, by, bw, bh = cv2.boundingRect(c)
            bbox = (int(bx * x_r), int(by * y_r), int(bw * x_r), int(bh * y_r))

            eventos.append(self._salvar_evento(f2, bbox))
            self.ultimo_salvamento = agora

        # Atualização segura de frames
        self.frame1 = self.frame2
        ret, f_new = self.cap.read()
        self.frame2 = f_new if ret else None

        return eventos

    def liberar(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()