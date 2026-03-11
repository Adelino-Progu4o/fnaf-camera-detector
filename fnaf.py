import numpy as np
import winsound

def aplicar_efeitos_fnaf(frame):
    """
    Mantém o frame colorido, opcionalmente com scanlines e ruído.
    """

    if frame is None:
        return None  # protege caso frame não exista

    final = frame.copy()

    # scanlines leves (opcional)
    for i in range(0, final.shape[0], 2):
        final[i,:,:] = (final[i,:,:] * 0.9).astype(np.uint8)

    # adicionar ruído sutil
    noise = np.random.randint(0, 10, (final.shape[0], final.shape[1], 3), dtype=np.uint8)
    final = np.clip(final + noise, 0, 255)

    return final

def tocar_alerta():
    """Toca beep curto para indicar movimento detectado."""
    winsound.Beep(1000, 150)