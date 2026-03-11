# alerta_gpio.py
import time

try:
    import RPi.GPIO as GPIO
    IS_PI = True
except (ImportError, RuntimeError):
    # fallback para PC sem GPIO
    IS_PI = False
    print("[SIMULAÇÃO GPIO] Nenhum GPIO disponível")

PIN_ALERTA = 18  # pino para LED ou buzzer

if IS_PI:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_ALERTA, GPIO.OUT)

def disparar_alerta(duration=0.5):
    """Acende LED/buzzer no Pi ou simula no PC"""
    if IS_PI:
        GPIO.output(PIN_ALERTA, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(PIN_ALERTA, GPIO.LOW)
    else:
        print(f"[SIMULAÇÃO GPIO] Alerta disparado por {duration} segundos")

def cleanup():
    """Limpa GPIO se estiver no Pi"""
    if IS_PI:
        GPIO.cleanup()