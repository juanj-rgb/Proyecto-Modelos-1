"""Errores previsibles de la aplicacion."""


class AplicacionError(Exception):
    """Error esperado: entrada invalida, archivo inexistente, version repetida, etc.

    Los scripts lo capturan, imprimen un mensaje claro en stderr y terminan
    con codigo de salida distinto de cero.
    """
