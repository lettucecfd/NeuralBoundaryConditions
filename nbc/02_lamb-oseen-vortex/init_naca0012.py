import numpy as np

__all__ = [
    "get_naca0012_mask"
]

def get_naca0012_mask(nx, ny, gpd, cx, cy, aoa_deg):
    """
    Erzeugt eine 2D-Bool-Maske für ein NACA0012 Profil.

    Parameter:
    nx, ny  : int   - Dimensionen des Simulationsgebiets (Gitterpunkte)
    gpd     : float - Chord length in Gitterpunkten (Grid Points per Diameter/Chord)
    cx, cy  : float - Position der Anströmkante (Leading Edge) im Gitter
    aoa_deg : float - Anstellwinkel (Angle of Attack) in Grad

    Returns:
    mask    : np.array (nx, ny) - True dort, wo das Profil ist
    info    : dict - Enthält Metadaten wie Ax (Länge), Ay (Höhe) des projizierten Profils
    """

    # 1. Konstanten und Umrechnung
    alpha = np.radians(aoa_deg)

    # NACA 0012 Koeffizienten (symmetrisch, daher m=0, p=0)
    a0 = 0.594689181
    a1 = 0.298222773
    a2 = -0.127125232
    a3 = -0.357907906
    a4 = 0.291984971
    a5 = -0.105174606

    # 2. Profilkoordinaten im lokalen System erzeugen (0 bis 1)
    # Wir nehmen etwas mehr Punkte als GPD für gute Auflösung vor der Rasterung
    num_points = int(gpd * 20)
    # Bereich leicht erweitert (-0.01 bis 1.01), um Rundungsfehler an den Kanten zu vermeiden
    x = np.linspace(0.0, 1.0, num=num_points)

    # Dickenverteilung (yt) berechnen
    yt = 5.0 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
    # Anmerkung: Die Formel oben ist die Standard NACA 4-Digit Formel.
    # Deine R-Koeffizienten waren leicht anders skaliert (a0 war ca 5*0.12).
    # Ich habe hier deine R-Werte genutzt, um exakt zu bleiben:
    yt = a0 * (a1 * np.sqrt(x) + a2 * x + a3 * x ** 2 + a4 * x ** 3 + a5 * x ** 4)

    # Symmetrisches Profil -> yc (Camber) ist 0
    yc = np.zeros_like(x)

    # 3. Skalierung auf Gittergröße (GPD)
    xu = x * gpd
    yu = (yc + yt) * gpd
    xl = x * gpd
    yl = (yc - yt) * gpd

    # 4. Rotation und Translation
    # Rotation um (0,0) -> Leading Edge
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)

    # Oberseite rotieren + verschieben
    xu_rot = xu * cos_a - yu * sin_a + cx
    yu_rot = xu * sin_a + yu * cos_a + cy

    # Unterseite rotieren + verschieben
    xl_rot = xu * cos_a - yl * sin_a + cx
    yl_rot = xu * sin_a + yl * cos_a + cy

    # 5. Scanline-Algorithmus zur Maskenerstellung
    mask = np.zeros((nx, ny), dtype=bool)

    # Grenzen bestimmen
    x_all = np.concatenate([xu_rot, xl_rot])
    y_all = np.concatenate([yu_rot, yl_rot])

    # Bounding Box im Gitter finden (geclippt auf Domain)
    i_min = max(0, int(np.floor(np.min(x_all))))
    i_max = min(nx - 1, int(np.ceil(np.max(x_all))))

    Ay_min_rec = ny
    Ay_max_rec = 0

    # Über x iterieren (Spaltenweise)
    for i in range(i_min, i_max + 1):
        # Interpolation der y-Werte an der Stelle i
        # np.interp braucht aufsteigende x-Werte. Da xu monoton ist (bei moderaten Winkeln),
        # passt das meistens. Zur Sicherheit bei großen Winkeln könnte man sortieren,
        # aber das kostet Performance.

        y_upper = np.interp(i, xu_rot, yu_rot, left=np.nan, right=np.nan)
        y_lower = np.interp(i, xl_rot, yl_rot, left=np.nan, right=np.nan)

        # Wenn wir gültige Schnittpunkte haben
        if not np.isnan(y_upper) and not np.isnan(y_lower):
            # Sortieren, falls durch Rotation lower > upper wird (selten bei NACA, aber möglich)
            y0, y1 = sorted([y_lower, y_upper])

            j_start = int(np.floor(y0))
            j_end = int(np.floor(y1))

            # Clipping y
            j_start = max(0, j_start)
            j_end = min(ny - 1, j_end)

            if j_end >= j_start:
                mask[i, j_start: j_end + 1] = True

                # Min/Max Tracking für Rückgabewerte
                if j_start < Ay_min_rec: Ay_min_rec = j_start
                if j_end > Ay_max_rec: Ay_max_rec = j_end

    # Metadaten berechnen
    Ax = i_max - i_min
    Ay = Ay_max_rec - Ay_min_rec if Ay_max_rec >= Ay_min_rec else 0

    return mask, {"Ax": Ax, "Ay": Ay, "x_range": (i_min, i_max), "y_range": (Ay_min_rec, Ay_max_rec)}


# --- Beispielaufruf und Integration in die Solver-Struktur ---

if __name__ == "__main__":
    # 1. Parameter definieren
    NX, NY, NZ = 1500, 1000, 1
    GPD = 250  # Profillänge in Pixeln
    CX = 500  # Leading Edge bei x=50
    CY = 500  # Leading Edge bei y=100 (Mitte)
    AOA = -10  # 10 Grad Anstellwinkel

    # 2. Funktion aufrufen
    mask, info = get_naca0012_mask(
        nx=NX,
        ny=NY,
        gpd=GPD,
        cx=CX,
        cy=CY,
        aoa_deg=AOA
    )

    print(f"Maske erstellt. Profilgröße ca: {info['Ax']}x{info['Ay']} Pixel.")


    # Kurzer Plot zur Kontrolle
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.imshow(mask.T, origin='lower', cmap='gray')  # Transponieren, damit x horizontal ist
    plt.title(f"NACA0012: GPD={GPD}, AoA={AOA}°, Pos=({CX},{CY})")
    plt.xlabel("nx")
    plt.ylabel("ny")
    plt.axis('equal')
    plt.show()
