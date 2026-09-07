import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

try:
    import sounddevice as sd
except (ImportError, OSError):
    # ImportError: el paquete no está instalado. OSError: está instalado pero
    # no encuentra la librería nativa PortAudio (típico en algunos entornos
    # Linux sin PortAudio del sistema). En ambos casos, la simulación pura
    # sigue funcionando sin esta dependencia; solo hace falta para
    # reproducir_trama()/grabar_recepcion()/reproducir_y_grabar() con hardware real.
    sd = None


class OFDMSystem:
    def __init__(self, N=64, CP_len=16, bits_per_symbol=6, N_datos=52, u_zc=25, semilla=None,
                 indices_piloto=(7, 21, 43, 57), valor_piloto=1.0 + 0j):
        if N_datos % 2 != 0:
            raise ValueError("N_datos debe ser par (se reparte a ambos lados del espectro)")
        if N_datos + 2 > N:
            raise ValueError("N_datos + DC + al menos 1 subportadora de guarda debe caber en N")

        self.N = N
        self.CP_len = CP_len
        self.bits_per_symbol = bits_per_symbol
        self.N_datos = N_datos
        self.u_zc = u_zc
        self.rng = np.random.default_rng(semilla)

        # Rejilla de constelación normalizada en energía media (BPSK o QAM cuadrada)
        self.grid = self._generar_rejilla(bits_per_symbol)

        # Pilotos: subportadoras dentro de las "usadas" (self.N_datos) que se
        # reservan con un valor conocido (valor_piloto) en vez de datos, para
        # poder estimar el canal H(k) en recepción. indices_piloto=() preserva
        # el comportamiento anterior (todas las subportadoras usadas son datos).
        self.indices_piloto = tuple(sorted(indices_piloto))
        self.valor_piloto = complex(valor_piloto)
        self.idx_piloto, self.idx_datos = self._calcular_indices_subportadoras()
        self.n_datos_reales = len(self.idx_datos)  # símbolos QAM de datos por símbolo OFDM

        # Pesos de interpolación lineal (en índice de subportadora) desde los
        # pilotos hacia las subportadoras de datos; se precalculan una única
        # vez porque las posiciones de piloto/datos no cambian entre símbolos.
        self._pesos_interp_pilotos = (
            self._matriz_interpolacion_pilotos() if len(self.idx_piloto) >= 2 else None
        )

    def _generar_rejilla(self, bits_per_symbol):
        if bits_per_symbol == 1:
            grid = np.array([-1.0, 1.0], dtype=complex)  # BPSK
        else:
            m = int(np.sqrt(2 ** bits_per_symbol))
            if m * m != 2 ** bits_per_symbol:
                raise ValueError(
                    "bits_per_symbol debe ser 1 (BPSK) o par: 2 (QPSK), 4 (16-QAM), 6 (64-QAM)..."
                )
            niveles = 2 * np.arange(m) - (m - 1)  # p.ej. m=4 -> [-3,-1,1,3]
            grid = np.array([x + 1j * y for y in niveles[::-1] for x in niveles])
        energia_media = np.mean(np.abs(grid) ** 2)
        return grid / np.sqrt(energia_media)

    def _calcular_indices_subportadoras(self):
        """Separa las self.N_datos subportadoras "usadas" (sin DC ni guarda) en piloto y datos."""
        half = self.N_datos // 2
        usadas = np.concatenate([np.arange(1, 1 + half), np.arange(self.N - half, self.N)])

        idx_piloto = np.array(self.indices_piloto, dtype=int)
        if len(idx_piloto) > 0 and not np.all(np.isin(idx_piloto, usadas)):
            raise ValueError(
                "indices_piloto debe estar contenido en las subportadoras usadas "
                f"(ni DC ni banda de guarda): {sorted(usadas.tolist())}"
            )

        idx_datos = np.array(sorted(set(usadas.tolist()) - set(idx_piloto.tolist())), dtype=int)
        return idx_piloto, idx_datos

    def _matriz_interpolacion_pilotos(self):
        """
        Precalcula los pesos de una interpolación lineal (en índice de
        subportadora k) desde self.idx_piloto hacia self.idx_datos, con
        extrapolación plana (clamp) fuera del rango de los pilotos extremos.
        Al ser fijos entre símbolos, permite estimar el canal de TODOS los
        bloques de una vez mediante un único producto matricial (sin bucle
        por símbolo) en 'estimar_canal_pilotos'.
        """
        pesos = np.zeros((len(self.idx_datos), len(self.idx_piloto)))
        for j, k in enumerate(self.idx_datos):
            if k <= self.idx_piloto[0]:
                pesos[j, 0] = 1.0
            elif k >= self.idx_piloto[-1]:
                pesos[j, -1] = 1.0
            else:
                i_sup = int(np.searchsorted(self.idx_piloto, k))
                i_inf = i_sup - 1
                k_inf, k_sup = self.idx_piloto[i_inf], self.idx_piloto[i_sup]
                t = (k - k_inf) / (k_sup - k_inf)
                pesos[j, i_inf] = 1 - t
                pesos[j, i_sup] = t
        return pesos

    # ------------------------------------------------------------------
    # 1. GENERACIÓN DE BITS Y MAPEO A SÍMBOLS
    # ------------------------------------------------------------------
    def generar_bits_y_mapeo(self, n_bits=None, bits=None):
        if bits is None:
            if n_bits is None:
                raise ValueError("Debes indicar n_bits o pasar un array de bits")
            bits_originales = self.rng.integers(0, 2, n_bits)
        else:
            bits_originales = np.asarray(bits, dtype=int)

        sobrantes = len(bits_originales) % self.bits_per_symbol
        if sobrantes == 0:
            bits_padded = bits_originales
        else:
            relleno = self.bits_per_symbol - sobrantes
            bits_padded = np.append(bits_originales, np.zeros(relleno, dtype=int))

        if self.bits_per_symbol == 1:
            indices = bits_padded
        else:
            bits_agrupados = bits_padded.reshape(-1, self.bits_per_symbol)  # (n_simbolos, bits_per_symbol)
            indices = bits_agrupados.dot(2 ** np.arange(self.bits_per_symbol)[::-1])

        simbolos = self.grid[indices]
        return bits_originales, simbolos

    # ------------------------------------------------------------------
    # 2. MODULACIÓN OFDM: mapeo de subportadoras + IFFT (vectorizada por bloques)
    # ------------------------------------------------------------------
    def modular_ofdm(self, simbolos):
        """
        Mapea símbolos QAM/BPSK a las subportadoras de DATOS (self.idx_datos)
        y, si hay pilotos configurados, fija self.valor_piloto en
        self.idx_piloto en TODOS los símbolos OFDM (mismo valor y fase
        conocidos, para poder estimar H(k) en recepción). La capacidad de
        datos por símbolo OFDM es self.n_datos_reales = N_datos - n_pilotos.
        """
        num_bloques = int(np.ceil(len(simbolos) / self.n_datos_reales))
        relleno = num_bloques * self.n_datos_reales - len(simbolos)
        simbolos_prep = np.append(simbolos, np.zeros(relleno, dtype=complex))
        chunks = simbolos_prep.reshape(num_bloques, self.n_datos_reales)  # (num_bloques, n_datos_reales)

        espectro = np.zeros((num_bloques, self.N), dtype=complex)  # (num_bloques, N)
        espectro[:, self.idx_datos] = chunks
        if len(self.idx_piloto) > 0:
            espectro[:, self.idx_piloto] = self.valor_piloto

        # IFFT fila a fila: cada bloque de frecuencia (N,) -> bloque de tiempo (N,)
        simbolos_tiempo = np.fft.ifft(espectro, axis=1)  # (num_bloques, N)
        return simbolos_tiempo

    # ------------------------------------------------------------------
    # 3. PREFIJO CÍCLICO
    # ------------------------------------------------------------------
    def añadir_prefijo_ciclico(self, simbolos_tiempo):
        prefijos = simbolos_tiempo[:, -self.CP_len:]
        return np.concatenate([prefijos, simbolos_tiempo], axis=1)

    def generar_preambulo(self):
        half = self.N_datos // 2
        n = np.arange(self.N_datos)
        zc = np.exp(-1j * np.pi * self.u_zc * n * (n + 1) / self.N_datos)

        espectro = np.zeros(self.N, dtype=complex)
        espectro[1:1 + half] = zc[:half]
        espectro[-half:] = zc[half:]

        simbolo_tiempo = np.fft.ifft(espectro)
        prefijo = simbolo_tiempo[-self.CP_len:]
        return np.concatenate([prefijo, simbolo_tiempo])

    def construir_trama(self, simbolos):
        simbolos_tiempo = self.modular_ofdm(simbolos)
        bloques_con_cp = self.añadir_prefijo_ciclico(simbolos_tiempo)  # (num_bloques, N+CP_len)
        datos_1d = bloques_con_cp.reshape(-1)
        preambulo = self.generar_preambulo()
        return np.concatenate([preambulo, datos_1d])

    # ------------------------------------------------------------------
    # 4. CANAL: AWGN (+ retraso opcional)
    # ------------------------------------------------------------------
    def canal_simulacion(self, senal, snr_db, retraso_muestras=0):
        if retraso_muestras > 0:
            senal = np.concatenate([np.zeros(retraso_muestras, dtype=complex), senal])

        potencia_senal = np.mean(np.abs(senal) ** 2)
        snr_lineal = 10 ** (snr_db / 10)
        potencia_ruido = potencia_senal / snr_lineal
        ruido = np.sqrt(potencia_ruido / 2) * (
            self.rng.standard_normal(len(senal)) + 1j * self.rng.standard_normal(len(senal))
        )
        return senal + ruido

    # ------------------------------------------------------------------
    # SINCRONIZACIÓN: correlación cruzada con el preámbulo
    # ------------------------------------------------------------------
    def sincronizar(self, senal_recibida):
        preambulo = self.generar_preambulo()
        correlacion = np.correlate(senal_recibida, preambulo, mode="valid")
        energia_preambulo = np.sqrt(np.sum(np.abs(preambulo) ** 2))
        correlacion_norm = np.abs(correlacion) / energia_preambulo

        inicio_preambulo = int(np.argmax(correlacion_norm))
        inicio_datos = inicio_preambulo + len(preambulo)
        return inicio_datos, inicio_preambulo, correlacion_norm

    # ------------------------------------------------------------------
    # SINCRONIZACIÓN MÚLTIPLE: localiza varias ráfagas dentro de una misma
    # grabación (varios preámbulos), en vez de un único pico global.
    #
    # Se usa cuando, en vez de transmitir una sola ráfaga corta, se envían
    # 'num_repeticiones' copias de la misma trama seguidas (con huecos de
    # silencio entre ellas) dentro de una única llamada a
    # 'reproducir_y_grabar' (sección 17). No sustituye a 'sincronizar': es
    # un método aparte, retrocompatible, para no romper su uso existente.
    # ------------------------------------------------------------------
    def sincronizar_multiples(self, senal_recibida, num_repeticiones, separacion_minima_muestras):
        """
        Encuentra hasta 'num_repeticiones' picos de correlación con el
        preámbulo, exigiendo una separación mínima entre ellos
        ('separacion_minima_muestras', típicamente la longitud completa de
        una ráfaga) para no confundir dos lóbulos secundarios del mismo
        preámbulo con dos ráfagas distintas. Devuelve las posiciones de
        inicio de preámbulo/datos en ORDEN TEMPORAL (no por altura de pico).
        """
        preambulo = self.generar_preambulo()
        correlacion = np.correlate(senal_recibida, preambulo, mode="valid")
        energia_preambulo = np.sqrt(np.sum(np.abs(preambulo) ** 2))
        correlacion_norm = np.abs(correlacion) / energia_preambulo

        picos_idx, _ = signal.find_peaks(correlacion_norm, distance=max(1, int(separacion_minima_muestras)))

        if len(picos_idx) == 0:
            return [], [], correlacion_norm

        # Nos quedamos con los 'num_repeticiones' picos más altos (más
        # fiables como preámbulos reales) y los reordenamos por tiempo,
        # que es el orden en que interesa procesarlos después.
        alturas = correlacion_norm[picos_idx]
        orden_por_altura = np.argsort(alturas)[::-1][:num_repeticiones]
        picos_seleccionados = np.sort(picos_idx[orden_por_altura])

        inicios_preambulo = [int(p) for p in picos_seleccionados]
        inicios_datos = [p + len(preambulo) for p in inicios_preambulo]
        return inicios_datos, inicios_preambulo, correlacion_norm

    # ------------------------------------------------------------------
    # 5. QUITAR PREFIJO CÍCLICO
    # ------------------------------------------------------------------
    def remover_prefijo_ciclico(self, senal_recibida, inicio_datos, num_bloques):
        longitud_bloque = self.N + self.CP_len
        num_bloques_disponibles = (len(senal_recibida) - inicio_datos) // longitud_bloque
        num_bloques = min(num_bloques, num_bloques_disponibles)

        fin = inicio_datos + num_bloques * longitud_bloque
        bloques_con_cp = senal_recibida[inicio_datos:fin].reshape(num_bloques, longitud_bloque)
        return bloques_con_cp[:, self.CP_len:]

    # ------------------------------------------------------------------
    # 6. DEMODULACIÓN OFDM: FFT + extracción de subportadoras
    # ------------------------------------------------------------------
    def demodular_ofdm(self, bloques_tiempo, canal_estimado=None):
        """
        bloques_tiempo: (num_bloques, N) sin CP.
        canal_estimado: opcional, admite forma (N,) -- una única corrección
        global aplicada por igual a todos los símbolos (uso original) -- o
        forma (num_bloques, N) -- una corrección DISTINTA por símbolo OFDM,
        p. ej. la combinación de CPE + ecualización por pilotos que arman
        'estimar_cpe_preambulo'/'estimar_canal_pilotos'. En ambos casos es
        una única división vectorizada (broadcasting de numpy), sin ruta
        paralela de ecualización.
        """
        espectro = np.fft.fft(bloques_tiempo, axis=1)  # (num_bloques, N)

        if canal_estimado is not None:
            espectro = espectro / canal_estimado

        datos = espectro[:, self.idx_datos]  # (num_bloques, n_datos_reales)
        return datos.reshape(-1)  # símbolos QAM recuperados, 1D

    # ------------------------------------------------------------------
    # 7. DEMAPEO + BER
    # ------------------------------------------------------------------
    def demapear(self, simbolos_rx):
        distancias = np.abs(simbolos_rx[:, None] - self.grid[None, :])  # (n_simbolos, tam_rejilla)
        indices = np.argmin(distancias, axis=1)

        if self.bits_per_symbol == 1:
            return indices.astype(int)

        shift_amounts = np.arange(self.bits_per_symbol - 1, -1, -1)
        bits = ((indices[:, None] >> shift_amounts) & 1).flatten()
        return bits

    def demapeo_y_ber(self, simbolos_rx, bits_originales):
        bits_rx = self.demapear(simbolos_rx)
        n_comparar = len(bits_originales)
        bits_rx_cortados = bits_rx[:n_comparar]

        errores = int(np.sum(bits_originales != bits_rx_cortados))
        ber = errores / n_comparar
        return ber, errores, bits_rx_cortados

    # ------------------------------------------------------------------
    # 8. CORRECCIÓN DE FASE COMÚN (CPE) A PARTIR DEL PREÁMBULO
    # ------------------------------------------------------------------
    def estimar_cpe_preambulo(self, senal_banda_base, inicio_preambulo):
        """
        Estima la rotación de fase COMÚN (CPE, Common Phase Error) a partir
        del preámbulo ya sincronizado, comparándolo con el preámbulo ideal
        conocido.

        Por qué separado de la ecualización por pilotos: el CPE lo provoca
        el retardo de propagación al mezclar/demezclar en pasobanda
        (e^{-j·2π·fc·τ}) y es una ÚNICA rotación que afecta por igual a
        TODAS las subportadoras (no depende de k); en cambio, un canal
        acústico multitrayecto real (selectividad en frecuencia) sí depende
        de k y varía lentamente símbolo a símbolo, por lo que se trata
        aparte en 'estimar_canal_pilotos'. Además, el preámbulo usa las
        self.N_datos (~52) subportadoras "usadas" en vez de solo los 4
        pilotos, por lo que promedia mucho más ruido y da una estimación de
        fase más precisa que si se intentara sacar el CPE de un único
        símbolo de datos con solo 4 puntos piloto.
        """
        preambulo_ideal_tiempo = self.generar_preambulo()
        longitud_preambulo = len(preambulo_ideal_tiempo)
        preambulo_recibido = senal_banda_base[inicio_preambulo: inicio_preambulo + longitud_preambulo]
        if len(preambulo_recibido) < longitud_preambulo:
            raise ValueError("La señal no contiene el preámbulo completo en 'inicio_preambulo'")

        # Reconstruye el espectro IDEAL del preámbulo (el mismo que arma
        # 'generar_preambulo' justo antes de la IFFT)
        half = self.N_datos // 2
        n = np.arange(self.N_datos)
        zc = np.exp(-1j * np.pi * self.u_zc * n * (n + 1) / self.N_datos)
        espectro_ideal = np.zeros(self.N, dtype=complex)
        espectro_ideal[1:1 + half] = zc[:half]
        espectro_ideal[-half:] = zc[half:]

        bloque_util_recibido = preambulo_recibido[self.CP_len:]
        espectro_recibido = np.fft.fft(bloque_util_recibido)

        usadas = np.concatenate([np.arange(1, 1 + half), np.arange(self.N - half, self.N)])
        razon = espectro_recibido[usadas] * np.conj(espectro_ideal[usadas])
        fase_comun = float(np.angle(np.sum(razon)))
        return fase_comun

    def corregir_cpe(self, espectro, fase_comun):
        """Aplica a 'espectro' (una fila (N,) o varias (num_bloques, N)) la corrección de fase estimada por 'estimar_cpe_preambulo'."""
        return espectro * np.exp(-1j * fase_comun)

    # ------------------------------------------------------------------
    # 9. ESTIMACIÓN DE CANAL POR PILOTOS (símbolo a símbolo)
    # ------------------------------------------------------------------
    def estimar_canal_pilotos(self, espectro):
        """
        Estima H(k) por SÍMBOLO OFDM (no una única estimación global) a
        partir de los pilotos conocidos, interpolando linealmente en
        índice de subportadora hacia las subportadoras de datos.

        Por qué símbolo a símbolo: un multitrayecto acústico de sala puede
        variar lentamente a lo largo de la ráfaga (movimiento del micrófono,
        reflexiones cambiantes); usar una H(k) distinta por símbolo permite
        seguir esa variación en vez de asumir un canal estático para toda
        la trama.

        'espectro' debe llegar ya corregido de CPE (ver 'corregir_cpe'), de
        forma que lo que capturan los pilotos aquí sea solo la selectividad
        en frecuencia del canal, no la rotación común de portadora.

        Devuelve un array (num_bloques, N) con H estimado en self.idx_datos
        y 1.0 en el resto de posiciones (no usadas, se ignoran después);
        listo para pasar como 'canal_estimado' a 'demodular_ofdm'.
        """
        if self._pesos_interp_pilotos is None:
            raise ValueError("Se necesitan al menos 2 pilotos configurados para poder interpolar en frecuencia")

        pilotos_rx = espectro[:, self.idx_piloto]              # (num_bloques, n_pilotos)
        h_en_pilotos = pilotos_rx / self.valor_piloto           # (num_bloques, n_pilotos)

        # Interpolación lineal vectorizada para TODOS los símbolos a la vez
        # (los pesos son los mismos en cada símbolo, solo cambian los valores medidos)
        h_en_datos = h_en_pilotos @ self._pesos_interp_pilotos.T  # (num_bloques, n_datos_reales)

        canal_estimado = np.ones((espectro.shape[0], self.N), dtype=complex)
        canal_estimado[:, self.idx_datos] = h_en_datos
        return canal_estimado

    # ------------------------------------------------------------------
    # NOTA SOBRE HARDWARE REAL (altavoz + micrófono):
    #
    # El reloj del DAC del altavoz y el del ADC del micrófono son osciladores
    # físicamente distintos y NO están sincronizados entre sí: además de un
    # retardo fijo (el que ya cubre 'sincronizar' vía el preámbulo), existe
    # una deriva de reloj (clock drift) que hace que el punto óptimo de
    # muestreo se desplace progresivamente a lo largo de la trama. Esta clase
    # NO implementa un lazo de seguimiento de fase por símbolo (p. ej. un PLL
    # de tiempo de símbolo); en su lugar, la mitigación práctica es transmitir
    # en RÁFAGAS CORTAS (unas pocas decenas/cientos de símbolos OFDM): en una
    # ráfaga corta, el desplazamiento acumulado por la deriva de reloj se
    # mantiene muy por debajo de la duración del prefijo cíclico, por lo que
    # una única sincronización al inicio de la ráfaga (la que ya hace
    # 'sincronizar') sigue siendo válida para todos los símbolos de esa
    # ráfaga. Ver PASO 5 del bloque __main__ para un ejemplo con varias
    # ráfagas cortas encadenadas pensado para esta misma limitación: cada
    # ráfaga se sincroniza y corrige de fase por separado, en vez de asumir
    # que una única estimación siga siendo válida varios segundos después.
    #
    # Tampoco se corrige el CFO (Carrier Frequency Offset, el desajuste entre
    # los osciladores de portadora de Tx y Rx): 'demodular_pasobanda' asume fc
    # exactamente conocida e idéntica en emisor y receptor. Un CFO real se
    # manifestaría como una rotación de fase que CRECE con el tiempo (a
    # diferencia del CPE, que es una rotación constante) y necesitaría un
    # lazo de seguimiento (p. ej. Costas loop) para corregirse correctamente;
    # queda fuera del alcance de esta fase, igual que el seguimiento de
    # deriva de reloj muestra a muestra.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 10. MODULACIÓN PASOBANDA (up-conversion IQ para transmitir por audio)
    # ------------------------------------------------------------------
    def modular_pasobanda(self, senal_banda_base, fc=8000.0, fs_audio=44100,
                           factor_sobremuestreo=20, ruta_wav=None, margen_clipping=0.9):
        """
        Convierte la envolvente compleja en banda base (salida de
        'construir_trama') en una señal real audible.

        Por qué: 'construir_trama' genera muestras complejas a la tasa
        "simbólica" del sistema OFDM (una muestra de IFFT por muestra), que
        no es directamente audible ni reproducible por un DAC de audio
        estándar. Aquí se sobremuestrea la señal (factor_sobremuestreo,
        equivalente a fijar la tasa de muestreo en banda base en
        fs_audio/factor_sobremuestreo) para acotar su ancho de banda, y se
        traslada en frecuencia a fc mezclando con una portadora compleja:
        Re{ x(t)·e^{j·2π·fc·t} } = I(t)·cos(2π·fc·t) − Q(t)·sin(2π·fc·t).
        El resultado es una señal real de doble banda lateral centrada en fc,
        que contiene toda la información de I/Q y es recuperable con
        'demodular_pasobanda'.

        senal_banda_base: array 1D complejo (p. ej. salida de construir_trama).
        Devuelve el array real (float) ya normalizado para evitar clipping,
        y opcionalmente lo guarda en 'ruta_wav' a fs_audio (44.1 kHz por defecto).
        """
        # Sobremuestreo con filtrado anti-imagen incluido (polyphase)
        senal_sobremuestreada = signal.resample_poly(senal_banda_base, factor_sobremuestreo, 1)

        n = np.arange(len(senal_sobremuestreada))
        t = n / fs_audio
        portadora = np.exp(1j * 2 * np.pi * fc * t)

        pasobanda = np.real(senal_sobremuestreada * portadora)  # (len(senal_sobremuestreada),) real

        pico = np.max(np.abs(pasobanda))
        if pico > 0:
            pasobanda = (pasobanda / pico) * margen_clipping  # normalización anti-clipping

        if ruta_wav is not None:
            wavfile.write(ruta_wav, fs_audio, (pasobanda * 32767).astype(np.int16))

        return pasobanda.astype(np.float64)

    # ------------------------------------------------------------------
    # 11. DEMODULACIÓN PASOBANDA (down-conversion IQ desde audio capturado)
    # ------------------------------------------------------------------
    def demodular_pasobanda(self, senal_audio, fc=8000.0, fs_audio=44100,
                             factor_sobremuestreo=20, orden_filtro=4):
        """
        Recupera la envolvente compleja en banda base a partir de audio real
        (grabación del micrófono), invirtiendo 'modular_pasobanda'.

        Mezcla con una portadora local coherente e^{-j·2π·fc·t} (se asume fc
        conocida y sin offset de frecuencia entre Tx/Rx; un desajuste de
        oscilador -CFO- no se corrige aquí) y filtra paso bajo con un
        Butterworth para eliminar la réplica en −2·fc y el ruido fuera de
        banda, antes de diezmar a la tasa "simbólica" original.

        Tras diezmar se normaliza la potencia media a 1 (AGC simple): la
        ganancia real del camino altavoz→aire→micrófono es desconocida y
        arbitraria (depende del volumen, la distancia, la sensibilidad del
        micrófono...), y además la propia demodulación pasobanda introduce
        un factor de escala fijo de 0.5. Sin esta normalización el demapeo
        por distancia mínima ('demapear') tomaría decisiones incorrectas
        aunque la forma de la constelación fuera correcta.
        """
        n = np.arange(len(senal_audio))
        t = n / fs_audio
        oscilador_local = np.exp(-1j * 2 * np.pi * fc * t)
        mezclado = senal_audio * oscilador_local  # banda base + réplica en -2*fc

        fs_banda_base = fs_audio / factor_sobremuestreo
        frecuencia_corte = fs_banda_base / 2  # Nyquist de la tasa de banda base objetivo
        b, a = signal.butter(orden_filtro, frecuencia_corte, fs=fs_audio, btype="low")
        mezclado_filtrado = signal.filtfilt(b, a, mezclado)

        senal_banda_base = mezclado_filtrado[::factor_sobremuestreo]  # diezmado

        potencia_media = np.mean(np.abs(senal_banda_base) ** 2)
        if potencia_media > 0:
            senal_banda_base = senal_banda_base / np.sqrt(potencia_media)  # AGC simple

        return senal_banda_base

    # ------------------------------------------------------------------
    # 12. REPRODUCCIÓN POR ALTAVOZ
    #
    # NOTA: si el altavoz y el micrófono son el MISMO dispositivo de audio
    # (caso típico de laboratorio con una única tarjeta/interfaz), usar
    # 'reproducir_y_grabar' (sección 17, sd.playrec) en vez de encadenar este
    # método con 'grabar_recepcion': dos streams separados (play + rec) no
    # garantizan que la grabación arranque exactamente cuando arranca la
    # reproducción, y con dos dispositivos DISTINTOS sí puede ser necesario
    # usar estos dos métodos por separado. Se conservan por eso, pero no son
    # el camino recomendado para un único dispositivo.
    # ------------------------------------------------------------------
    def reproducir_trama(self, senal_audio, fs_audio=44100, bloqueante=True):
        """Reproduce por el altavoz predeterminado la señal real (salida de modular_pasobanda)."""
        if sd is None:
            raise ImportError("Se necesita 'sounddevice' (pip install sounddevice) para reproducir audio.")
        sd.play(senal_audio.astype(np.float32), samplerate=fs_audio, blocking=bloqueante)

    # ------------------------------------------------------------------
    # 13. GRABACIÓN POR MICRÓFONO
    # ------------------------------------------------------------------
    def grabar_recepcion(self, duracion_s, fs_audio=44100, canales=1):
        """
        Graba por el micrófono predeterminado durante 'duracion_s' segundos.
        'duracion_s' debe cubrir la duración de la trama transmitida más un
        margen (algunas décimas de segundo) para la latencia de arranque de
        los buffers de audio del sistema operativo, de forma que el preámbulo
        no quede cortado al principio o al final de la grabación.
        """
        if sd is None:
            raise ImportError("Se necesita 'sounddevice' (pip install sounddevice) para grabar audio.")
        grabacion = sd.rec(int(duracion_s * fs_audio), samplerate=fs_audio,
                            channels=canales, dtype="float64", blocking=True)
        return grabacion.flatten()

    # ------------------------------------------------------------------
    # 14. RECEPTOR COMPLETO ENCADENADO (pensado para audio real de micrófono,
    #     UNA sola ráfaga)
    #
    # sincronización -> demodulación pasobanda -> FFT -> corrección CPE ->
    # ecualización por pilotos -> demapeo -> BER. Reutiliza exactamente los
    # métodos anteriores (ninguna lógica de ecualización/extracción paralela):
    # la corrección de CPE y la de pilotos se combinan en un único array
    # 'canal_estimado' que se pasa a 'demodular_ofdm', igual que en la
    # comparativa del PASO 6 del bloque __main__.
    #
    # Para una grabación con VARIAS ráfagas encadenadas, ver
    # 'procesar_multiples_rafagas' (sección 16), que aplica esta misma
    # lógica de forma independiente a cada ráfaga detectada.
    # ------------------------------------------------------------------
    def receptor_completo(self, grabacion_audio, bits_originales, num_bloques,
                           fc=8000.0, fs_audio=44100, factor_sobremuestreo=20):
        """
        Procesa una grabación de audio real (salida de 'grabar_recepcion')
        de principio a fin y devuelve (ber, errores, simbolos_recuperados,
        diagnostico). 'diagnostico' incluye el punto de sincronización y la
        fase común (CPE) estimada, útiles para depurar en el laboratorio.
        """
        senal_banda_base = self.demodular_pasobanda(
            grabacion_audio, fc=fc, fs_audio=fs_audio, factor_sobremuestreo=factor_sobremuestreo
        )
        inicio_datos, inicio_preambulo, correlacion = self.sincronizar(senal_banda_base)
        fase_comun = self.estimar_cpe_preambulo(senal_banda_base, inicio_preambulo)

        bloques_tiempo = self.remover_prefijo_ciclico(senal_banda_base, inicio_datos, num_bloques)

        if len(self.idx_piloto) >= 2:
            espectro_cpe = self.corregir_cpe(np.fft.fft(bloques_tiempo, axis=1), fase_comun)
            canal_estimado = self.estimar_canal_pilotos(espectro_cpe) * np.exp(1j * fase_comun)
        else:
            # Sin pilotos suficientes: solo se corrige el CPE (constante en toda la subportadora)
            canal_estimado = np.exp(1j * fase_comun) * np.ones(self.N, dtype=complex)

        simbolos_recuperados = self.demodular_ofdm(bloques_tiempo, canal_estimado=canal_estimado)
        ber, errores, _ = self.demapeo_y_ber(simbolos_recuperados, bits_originales)

        diagnostico = {
            "inicio_preambulo": inicio_preambulo,
            "inicio_datos": inicio_datos,
            "fase_comun_grados": float(np.degrees(fase_comun)),
            "correlacion": correlacion,
        }
        return ber, errores, simbolos_recuperados, diagnostico

    # ------------------------------------------------------------------
    # 15. REPRODUCCIÓN + GRABACIÓN SIMULTÁNEA (un único dispositivo, mismo reloj)
    #
    # Camino recomendado cuando el altavoz y el micrófono son el MISMO
    # dispositivo de audio: sd.playrec() reproduce y graba en un único
    # stream, así que ambas operaciones comparten el mismo reloj de muestreo
    # y arrancan exactamente a la vez (a diferencia de encadenar
    # 'grabar_recepcion' + 'reproducir_trama' por separado, dos streams
    # independientes sin garantía de arranque simultáneo). La duración de
    # entrada y la de la grabación son, por definición, la misma: el margen
    # de silencio para cubrir la latencia de los buffers debe venir ya
    # incluido en 'senal_audio' (ver PASO 5 del bloque __main__).
    # ------------------------------------------------------------------
    def reproducir_y_grabar(self, senal_audio, fs_audio=44100, canales=1):
        """
        Reproduce 'senal_audio' por el altavoz y graba por el micrófono
        SIMULTÁNEAMENTE en un único stream (sd.playrec). Devuelve la
        grabación como array 1D real de la misma longitud que 'senal_audio'.
        """
        if sd is None:
            raise ImportError("Se necesita 'sounddevice' (pip install sounddevice) para reproducir/grabar audio.")
        grabacion = sd.playrec(
            senal_audio.astype(np.float32), samplerate=fs_audio, channels=canales, blocking=True
        )
        return grabacion.flatten()

    # ------------------------------------------------------------------
    # 16. PROCESAMIENTO DE VARIAS RÁFAGAS ENCADENADAS (misma trama repetida
    #     'num_repeticiones' veces dentro de UNA sola grabación)
    #
    # Cada ráfaga se sincroniza, corrige de CPE, ecualiza por pilotos,
    # demapea y evalúa de forma INDEPENDIENTE, exactamente igual que si cada
    # una fuera el resultado de una ejecución aislada de 'receptor_completo'
    # (sección 14) — no se reutiliza el CPE ni el canal estimado de una
    # ráfaga para otra. Esto preserva la independencia estadística entre
    # repeticiones que exige el protocolo experimental (Capítulo 4 de la
    # memoria: 10 repeticiones por distancia), a diferencia de transmitir
    # una única ráfaga larga continua, donde una sola estimación de CPE se
    # aplicaría a toda la trama.
    # ------------------------------------------------------------------
    def procesar_multiples_rafagas(self, grabacion_audio, bits_originales_por_rafaga, num_bloques_por_rafaga,
                                    num_repeticiones, longitud_rafaga_muestras_bb,
                                    fc=8000.0, fs_audio=44100, factor_sobremuestreo=20):
        """
        grabacion_audio: array real, salida de 'reproducir_y_grabar', con
            'num_repeticiones' copias de la misma trama (separadas por
            huecos de silencio) dentro de una única grabación.
        bits_originales_por_rafaga: los bits transmitidos en CADA ráfaga
            (la misma secuencia se repite en las 'num_repeticiones' copias).
        num_bloques_por_rafaga: número de símbolos OFDM de datos por ráfaga
            (el mismo para todas, ya que es la misma trama repetida).
        longitud_rafaga_muestras_bb: longitud de una ráfaga completa
            (preámbulo + datos) en muestras de BANDA BASE, usada como
            separación mínima entre picos en 'sincronizar_multiples' para
            no detectar dos veces el mismo preámbulo.

        Devuelve una lista de diccionarios (uno por ráfaga detectada, en
        orden temporal), cada uno con: repeticion, pico, fase_comun_grados,
        errores, ber.
        """
        senal_banda_base = self.demodular_pasobanda(
            grabacion_audio, fc=fc, fs_audio=fs_audio, factor_sobremuestreo=factor_sobremuestreo
        )
        inicios_datos, inicios_preambulo, _ = self.sincronizar_multiples(
            senal_banda_base, num_repeticiones, separacion_minima_muestras=longitud_rafaga_muestras_bb
        )

        resultados = []
        for rep, (inicio_datos, inicio_preambulo) in enumerate(zip(inicios_datos, inicios_preambulo), start=1):
            fase_comun = self.estimar_cpe_preambulo(senal_banda_base, inicio_preambulo)
            bloques_tiempo = self.remover_prefijo_ciclico(senal_banda_base, inicio_datos, num_bloques_por_rafaga)

            if len(self.idx_piloto) >= 2:
                espectro_cpe = self.corregir_cpe(np.fft.fft(bloques_tiempo, axis=1), fase_comun)
                canal_estimado = self.estimar_canal_pilotos(espectro_cpe) * np.exp(1j * fase_comun)
            else:
                canal_estimado = np.exp(1j * fase_comun) * np.ones(self.N, dtype=complex)

            simbolos_recuperados = self.demodular_ofdm(bloques_tiempo, canal_estimado=canal_estimado)
            ber, errores, _ = self.demapeo_y_ber(simbolos_recuperados, bits_originales_por_rafaga)

            # Pico de amplitud de ESTA ráfaga en particular, sobre la señal
            # de audio cruda (no en banda base), para diagnóstico de nivel.
            # 'demodular_pasobanda' filtra con filtfilt (fase cero, sin
            # retardo), así que el índice en banda base multiplicado por
            # 'factor_sobremuestreo' es una aproximación razonable de la
            # posición correspondiente en la señal de audio original.
            idx_audio_inicio = max(0, inicio_preambulo * factor_sobremuestreo)
            idx_audio_fin = min(len(grabacion_audio),
                                 idx_audio_inicio + longitud_rafaga_muestras_bb * factor_sobremuestreo)
            segmento_audio = grabacion_audio[idx_audio_inicio:idx_audio_fin]
            pico_rafaga = float(np.max(np.abs(segmento_audio))) if len(segmento_audio) > 0 else float("nan")

            resultados.append({
                "repeticion": rep,
                "pico": pico_rafaga,
                "fase_comun_grados": float(np.degrees(fase_comun)),
                "errores": errores,
                "ber": ber,
            })

        return resultados

    # ------------------------------------------------------------------
    # 17. GUARDAR RESULTADOS DE MÚLTIPLES RÁFAGAS EN CSV (acumulativo)
    # ------------------------------------------------------------------
    def guardar_resultados_csv(self, resultados, ruta_csv="resultados_paso5.csv"):
        """
        Añade filas nuevas a 'ruta_csv' (una por repetición procesada por
        'procesar_multiples_rafagas'), escribiendo la cabecera solo si el
        archivo no existe todavía, para poder acumular varias sesiones de
        pruebas (distintas distancias, por ejemplo) sin perder las
        anteriores.
        """
        existe = os.path.isfile(ruta_csv)
        with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
            campos = ["Repeticion", "Pico", "Fase_CPE_grados", "Errores_bits", "BER"]
            escritor = csv.DictWriter(f, fieldnames=campos)
            if not existe:
                escritor.writeheader()
            for r in resultados:
                escritor.writerow({
                    "Repeticion": r["repeticion"],
                    "Pico": f"{r['pico']:.6f}",
                    "Fase_CPE_grados": f"{r['fase_comun_grados']:.1f}",
                    "Errores_bits": r["errores"],
                    "BER": f"{r['ber']:.6e}",
                })


# ------------------------------------------------------------------
# BLOQUE PRINCIPAL: EJECUCIÓN VISUAL PASO A PASO
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --- Parámetros del Sistema ---
    N = 64
    CP_LEN = 16
    BITS_PER_SYMBOL = 6 # 64-QAM: la modulación más exigente que soporta el sistema
    N_DATOS = 52
    N_BITS = 416000        # Número de bits grande, para tener una muestra estadística amplia en simulación
    SNR_DB = 18         # SNR de laboratorio simulado
    RETRASO_MUESTRAS = 150 

    print("="*60)
    print(" INICIANDO LABORATORIO OFDM PASO A PASO para TFG ")
    print("="*60)
    print(f"Configuración: FFT={N} | CP={CP_LEN} | Modulación={2**BITS_PER_SYMBOL}-QAM")
    
    ofdm = OFDMSystem(N=N, CP_len=CP_LEN, bits_per_symbol=BITS_PER_SYMBOL, N_datos=N_DATOS, semilla=42)

    # ------------------------------------------------------------------
    # PASO 1: Transmisor (Mapeo de Constelación)
    # ------------------------------------------------------------------
    bits_originales, simbolos_tx = ofdm.generar_bits_y_mapeo(n_bits=N_BITS)
    # OJO: la capacidad de datos por símbolo OFDM es ofdm.n_datos_reales
    # (N_DATOS menos los pilotos), no N_DATOS directamente.
    num_bloques = int(np.ceil(len(simbolos_tx) / ofdm.n_datos_reales))

    plt.figure(figsize=(6, 5))
    plt.scatter(simbolos_tx.real, simbolos_tx.imag, color="blue", marker="o", s=25, label="Símbolos Tx")
    plt.title("Paso 1: Símbolos Generados en el Transmisor (Ideal)")
    plt.grid(True, alpha=0.3)
    plt.xlabel("En Fase (I)")
    plt.ylabel("Cuadratura (Q)")
    plt.legend()
    print("\n[Paso 1/4] Mostrando constelación en el transmisor. Cierra la ventana gráfica para avanzar.")
    plt.show()

    # ------------------------------------------------------------------
    # PASO 2: Construcción de la Trama y Modulación en el Tiempo
    # ------------------------------------------------------------------
    senal_transmitida = ofdm.construir_trama(simbolos_tx)

    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(senal_transmitida), color="darkorange", lw=1)
    # Dibujar una línea vertical para marcar el fin del preámbulo
    longitud_preambulo = N + CP_LEN
    plt.axvline(x=longitud_preambulo, color="red", linestyle="--", label="Fin del Preámbulo / Inicio Datos")
    plt.title("Paso 2: Trama OFDM en el Dominio del Tiempo (Amplitud)")
    plt.xlabel("Muestras")
    plt.ylabel("|x(t)|")
    plt.grid(True, alpha=0.3)
    plt.legend()
    print("[Paso 2/4] Mostrando la trama temporal generada (Preámbulo + Datos). Cierra la gráfica para continuar.")
    plt.show()

    # ------------------------------------------------------------------
    # PASO 3: Canal y Sincronización en el Receptor
    # ------------------------------------------------------------------
    senal_recibida = ofdm.canal_simulacion(senal_transmitida, snr_db=SNR_DB, retraso_muestras=RETRASO_MUESTRAS)
    inicio_datos, inicio_preambulo, correlacion_norm = ofdm.sincronizar(senal_recibida)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(np.abs(senal_recibida), color="purple", alpha=0.7)
    ax1.set_title("Paso 3: Señal Recibida en el Tiempo (Con Ruido y Retraso)")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True, alpha=0.3)

    ax2.plot(correlacion_norm, color="green", lw=1.5)
    ax2.axvline(x=inicio_preambulo, color="red", linestyle=":", label=f"Pico detectado en muestra {inicio_preambulo}")
    ax2.set_title("Salida del Sincronizador (Correlación Cruzada con Zadoff-Chu)")
    ax2.set_xlabel("Muestras")
    ax2.set_ylabel("Magnitud Correlación")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    print(f"\n[Sincronizador] Retraso físico real: {RETRASO_MUESTRAS} muestras.")
    print(f"[Sincronizador] Pico estimado mediante Zadoff-Chu: {inicio_preambulo} muestras.")
    print("[Paso 3/4] Mostrando sincronización por correlación. Cierra la gráfica para extraer datos.")
    plt.show()

    # ------------------------------------------------------------------
    # PASO 4: Receptor (FFT, Extracción de Símbolos y BER)
    # ------------------------------------------------------------------
    bloques_tiempo = ofdm.remover_prefijo_ciclico(senal_recibida, inicio_datos, num_bloques)
    simbolos_recuperados = ofdm.demodular_ofdm(bloques_tiempo)
    ber, errores, _ = ofdm.demapeo_y_ber(simbolos_recuperados, bits_originales)

    print("\n"+"="*60)
    print(" RESULTADOS FINALES DE LA SIMULACIÓN DE CANAL ")
    print("="*60)
    print(f"-> Total de Bits Transmitidos : {len(bits_originales)}")
    print(f"-> Bits Erróneos Detectados  : {errores}")
    print(f"-> Tasa de Error de Bits (BER): {ber:.5e} (u {ber*100:.3f}%)")
    print("="*60)

    plt.figure(figsize=(6, 6))
    plt.scatter(simbolos_recuperados.real, simbolos_recuperados.imag, s=12, alpha=0.6, color="tab:red", label="Símbolos Rx")
    plt.scatter(ofdm.grid.real, ofdm.grid.imag, color="black", marker="x", s=40, lw=1.5, label="Constelación Ideal")
    plt.title(f"Paso 4: Constelación Recibida Final\nBER = {ber:.5e} | SNR = {SNR_DB} dB")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    print("\n[Paso 4/4] Mostrando constelación final degradada por el canal. Final del flujo.")
    plt.show()

    # ------------------------------------------------------------------
    # PASO 5 (OPCIONAL): Transmisión y recepción REALES por altavoz/micrófono
    #
    # Ancho de banda de trabajo ajustado a 3-7 kHz (fc=5000 Hz,
    # factor_sobremuestreo=9), zona donde el altavoz/micrófono disponibles
    # responden mejor (indicación del tutor).
    #
    # En vez de una única ráfaga larga continua, se transmiten
    # NUM_REPETICIONES copias de la MISMA ráfaga corta de 300 bits, cada
    # una con su propio preámbulo, separadas por un hueco corto de silencio
    # (GAP_ENTRE_RAFAGAS_S) para evitar que la reverberación acústica de una
    # ráfaga contamine el preámbulo de la siguiente y dar tiempo de
    # asentamiento al filtro paso bajo de 'demodular_pasobanda'. El margen
    # grande (0.5s) de latencia de arranque de los buffers de audio solo se
    # aplica una vez, al principio y al final de TODA la secuencia — no
    # entre cada ráfaga. Cada ráfaga se sincroniza y corrige de CPE por
    # separado (ver 'procesar_multiples_rafagas', sección 16), preservando
    # la independencia estadística entre repeticiones que exige el
    # protocolo experimental de la memoria (Capítulo 4: 10 repeticiones por
    # distancia).
    # ------------------------------------------------------------------
    RESPUESTA = input("\n¿Ejecutar también la demo con audio real (altavoz+micrófono)? [s/N]: ").strip().lower()
    if RESPUESTA == "s":
        if sd is None:
            print("Falta 'sounddevice' (pip install sounddevice); se omite la demo de audio real.")
        else:
            FS_AUDIO = 44100
            FC = 5000.0
            FACTOR_SOBREMUESTREO = 9
            N_BITS_RAFAGA_CORTA = 300
            NUM_REPETICIONES = 10
            GAP_ENTRE_RAFAGAS_S = 0.08  # 80 ms de silencio entre ráfagas consecutivas

            ofdm_audio = OFDMSystem(N=N, CP_len=CP_LEN, bits_per_symbol=BITS_PER_SYMBOL,
                                     N_datos=N_DATOS, semilla=123)
            bits_audio_tx, simbolos_audio_tx = ofdm_audio.generar_bits_y_mapeo(n_bits=N_BITS_RAFAGA_CORTA)
            num_bloques_audio = int(np.ceil(len(simbolos_audio_tx) / ofdm_audio.n_datos_reales))

            senal_banda_base_tx = ofdm_audio.construir_trama(simbolos_audio_tx)
            longitud_rafaga_muestras_bb = len(senal_banda_base_tx)  # preámbulo + datos, en banda base

            senal_pasobanda_una_rafaga = ofdm_audio.modular_pasobanda(
                senal_banda_base_tx, fc=FC, fs_audio=FS_AUDIO,
                factor_sobremuestreo=FACTOR_SOBREMUESTREO, ruta_wav="trama_audio_tx.wav"
            )

            gap_muestras_audio = int(GAP_ENTRE_RAFAGAS_S * FS_AUDIO)
            gap_silencio = np.zeros(gap_muestras_audio, dtype=senal_pasobanda_una_rafaga.dtype)

            # Concatena NUM_REPETICIONES copias de la ráfaga, con un hueco
            # corto entre ellas (no después de la última).
            partes = []
            for i in range(NUM_REPETICIONES):
                partes.append(senal_pasobanda_una_rafaga)
                if i < NUM_REPETICIONES - 1:
                    partes.append(gap_silencio)
            secuencia_rafagas = np.concatenate(partes)

            # Margen grande de latencia de arranque de buffers de audio,
            # una sola vez al principio y al final de TODA la secuencia.
            margen_latencia_s = 0.5
            margen_muestras = int(margen_latencia_s * FS_AUDIO)
            senal_con_margen = np.concatenate([
                np.zeros(margen_muestras, dtype=secuencia_rafagas.dtype),
                secuencia_rafagas,
                np.zeros(margen_muestras, dtype=secuencia_rafagas.dtype),
            ])
            duracion_total_s = len(senal_con_margen) / FS_AUDIO

            print(f"\n[Paso 5] Reproduciendo y grabando simultáneamente {NUM_REPETICIONES} ráfagas "
                  f"de {N_BITS_RAFAGA_CORTA} bits ({duracion_total_s:.2f} s totales, "
                  f"{margen_latencia_s:.1f} s de silencio en cada extremo, "
                  f"{GAP_ENTRE_RAFAGAS_S*1000:.0f} ms entre ráfagas)...")
            grabacion = ofdm_audio.reproducir_y_grabar(senal_con_margen, fs_audio=FS_AUDIO)

            resultados_rafagas = ofdm_audio.procesar_multiples_rafagas(
                grabacion, bits_audio_tx, num_bloques_audio,
                NUM_REPETICIONES, longitud_rafaga_muestras_bb,
                fc=FC, fs_audio=FS_AUDIO, factor_sobremuestreo=FACTOR_SOBREMUESTREO
            )

            print("\n" + "=" * 60)
            print(" RESULTADOS DEMO AUDIO REAL (ALTAVOZ + MICRÓFONO) — MÚLTIPLES RÁFAGAS ")
            print("=" * 60)
            if len(resultados_rafagas) < NUM_REPETICIONES:
                print(f"[Aviso] Solo se detectaron {len(resultados_rafagas)} de {NUM_REPETICIONES} ráfagas "
                      f"esperadas. Revisa el volumen/ganancia si el número es mucho menor de 10.")
            print(f"{'Rep':>4}{'Pico':>12}{'Fase CPE (°)':>15}{'Errores':>10}{'BER':>14}")
            for r in resultados_rafagas:
                print(f"{r['repeticion']:>4}{r['pico']:>12.4f}{r['fase_comun_grados']:>15.1f}"
                      f"{r['errores']:>10}{r['ber']:>14.4e}")
            print("=" * 60)

            for r in resultados_rafagas:
                if r["pico"] < 0.02:
                    print(f"[Aviso] Repetición {r['repeticion']}: pico muy bajo ({r['pico']:.4f}).")
                elif r["pico"] > 0.98:
                    print(f"[Aviso] Repetición {r['repeticion']}: pico muy alto ({r['pico']:.4f}), posible clipping.")

            ofdm_audio.guardar_resultados_csv(resultados_rafagas, ruta_csv="resultados_paso5.csv")
            print("\n[Paso 5] Resultados añadidos a 'resultados_paso5.csv' "
                  "(columnas: Repeticion, Pico, Fase_CPE_grados, Errores_bits, BER).")

            if resultados_rafagas:
                bers = [r["ber"] for r in resultados_rafagas]
                plt.figure(figsize=(8, 4))
                plt.plot(range(1, len(bers) + 1), bers, marker="o")
                plt.title(f"Paso 5: BER por repetición ({len(bers)} ráfagas)")
                plt.xlabel("Repetición")
                plt.ylabel("BER")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

    # ------------------------------------------------------------------
    # PASO 6: Tabla comparativa BER -- con/sin CPE, con/sin ecualización por pilotos
    #
    # 'canal_simulacion' es un canal AWGN puro en banda base: no modela la
    # rotación de fase de portadora (eso solo aparece al pasar por pasobanda,
    # ver PASO 5) ni la selectividad en frecuencia de un multitrayecto
    # acústico real. Para poder comparar aquí, en simulación pura y sin
    # depender de hardware, se inyectan manualmente ambos efectos sobre el
    # espectro ya recibido de PASO 3: una rotación de fase fija (imitando el
    # CPE) y un rizado suave en frecuencia (imitando el multitrayecto de una
    # sala), reutilizando exactamente 'demodular_ofdm'/'demapeo_y_ber' para
    # medir el efecto de cada corrección por separado.
    # ------------------------------------------------------------------
    FASE_CPE_SIMULADA_GRADOS = 65.0
    fase_cpe_simulada = np.deg2rad(FASE_CPE_SIMULADA_GRADOS)

    # Un único ciclo a lo largo de toda la banda (periodo=N, mucho más ancho
    # que la separación entre pilotos ~14-22 bins): un multitrayecto acústico
    # real varía lentamente en frecuencia, y así la interpolación lineal
    # entre los 4 pilotos puede seguirlo con fidelidad razonable.
    k_subportadora = np.arange(N)
    canal_sintetico = 1.0 + 0.4 * np.exp(-1j * 2 * np.pi * k_subportadora / N)

    longitud_preambulo = N + CP_LEN
    senal_recibida_demo = senal_recibida.copy()
    # Rota también el preámbulo recibido con la MISMA fase, para que
    # 'estimar_cpe_preambulo' vea una situación físicamente consistente
    # (en pasobanda real, preámbulo y datos sufren el mismo CPE porque
    # ambos atraviesan el mismo retardo de propagación).
    senal_recibida_demo[inicio_preambulo:inicio_preambulo + longitud_preambulo] *= np.exp(1j * fase_cpe_simulada)

    bloques_tiempo_demo = ofdm.remover_prefijo_ciclico(senal_recibida, inicio_datos, num_bloques)
    espectro_demo = np.fft.fft(bloques_tiempo_demo, axis=1)
    espectro_distorsionado = espectro_demo * canal_sintetico[None, :] * np.exp(1j * fase_cpe_simulada)
    bloques_tiempo_distorsionados = np.fft.ifft(espectro_distorsionado, axis=1)

    resultados_tabla = []

    # (a) Sin ninguna corrección
    simb_a = ofdm.demodular_ofdm(bloques_tiempo_distorsionados, canal_estimado=None)
    ber_a, err_a, _ = ofdm.demapeo_y_ber(simb_a, bits_originales)
    resultados_tabla.append(("Sin CPE, sin pilotos", ber_a, err_a))

    # (b) Solo corrección de CPE, estimada del preámbulo (sin pilotos)
    fase_estimada = ofdm.estimar_cpe_preambulo(senal_recibida_demo, inicio_preambulo)
    espectro_cpe_b = ofdm.corregir_cpe(espectro_distorsionado, fase_estimada)
    bloques_b = np.fft.ifft(espectro_cpe_b, axis=1)
    simb_b = ofdm.demodular_ofdm(bloques_b, canal_estimado=None)
    ber_b, err_b, _ = ofdm.demapeo_y_ber(simb_b, bits_originales)
    resultados_tabla.append(("Solo CPE (preámbulo)", ber_b, err_b))

    # (c) CPE + ecualización por pilotos (símbolo a símbolo)
    canal_pilotos_c = ofdm.estimar_canal_pilotos(espectro_cpe_b)
    canal_total_c = canal_pilotos_c * np.exp(1j * fase_estimada)
    simb_c = ofdm.demodular_ofdm(bloques_tiempo_distorsionados, canal_estimado=canal_total_c)
    ber_c, err_c, _ = ofdm.demapeo_y_ber(simb_c, bits_originales)
    resultados_tabla.append(("CPE + pilotos", ber_c, err_c))

    print("\n" + "=" * 60)
    print(" PASO 6: TABLA COMPARATIVA -- CPE / ECUALIZACIÓN POR PILOTOS ")
    print(f" (CPE simulado inyectado: {FASE_CPE_SIMULADA_GRADOS}°, fase estimada: {np.degrees(fase_estimada):.1f}°)")
    print("=" * 60)
    print(f"{'Configuración':<28}{'BER':>12}{'Errores':>12}")
    for nombre, ber_tabla, err_tabla in resultados_tabla:
        print(f"{nombre:<28}{ber_tabla:>12.4e}{err_tabla:>12d}")
    print("=" * 60)
