from ctypes.macholib import dyld
from tkinter import END
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import show

from domain.beatmap import Note, Obstacle, BSMap, Bomb
from domain.geometry import (
    dir_to_angle_rad,
    dir_to_angle_deg,
    bsnormalize,
    dir_to_vector_3d,
)
from typing import Dict, Any, List, Optional


class BSMapProfiler:

    def __init__(self, map: BSMap):
        self.map = map

    def compute_stats(self):
        # Calcul des statistiques de la map
        self.stats = {
            "map_name": self.map.name,
            "version": self.map.version,
            "bpm": self.map.bpm,
            "notes_count": len(self.map.notes),
            "obstacles_count": len(self.map.obstacles) if self.map.obstacles else 0,
            "bombs_count": len(self.map.bombs) if self.map.bombs else 0,
        }

        if self.map.duration:
            self.stats["duration_seconds"] = self.map.duration.get("seconds", 0)
            self.stats["duration_minutes"] = self.map.duration.get("minutes", 0)
            self.stats["duration_beats"] = self.map.duration.get("beats", 0)

        print("Map statistics:", self.stats)

    def compute_visualization_data(self):

        if not self.map.notes:
            print("No notes to visualize")
            return

        # Initialisation des listes pour les coordonn�es
        self.Xr, self.Yr, self.Zr = [], [], []
        self.Xb, self.Yb, self.Zb = [], [], []
        self.U_r, self.V_r, self.W_r = (
            [],
            [],
            [],
        )  # Pour les vecteurs de direction rouges
        self.U_b, self.V_b, self.W_b = (
            [],
            [],
            [],
        )  # Pour les vecteurs de direction bleus
        vector_length = 0.6

        # Plotly 3D scatter plot
        for note in self.map.notes:
            # Extraction des coordonn�es et des couleurs
            color = ["red" if note.saber == 0 else "blue"]
            cut = note.dir
            vector = dir_to_vector_3d(cut)
            if vector is None:
                print(
                    f"Invalid cut direction {cut} for note at time {note.time}, skipping"
                )
                continue
            dx, dy, dz = bsnormalize(vector)

            if color == ["red"]:
                self.Xr.append(note.col)
                self.Yr.append(note.time * 60 / self.map.bpm)
                self.Zr.append(note.row)
                self.U_r.append(dx)
                self.V_r.append(dy)
                self.W_r.append(dz)

            elif color == ["blue"]:
                self.Xb.append(note.col)
                self.Yb.append(note.time * 60 / self.map.bpm)
                self.Zb.append(note.row)
                self.U_b.append(dx)
                self.V_b.append(dy)
                self.W_b.append(dz)

        # Conversion des listes en tableaux numpy pour Plotly
        self.Xr = np.asarray(self.Xr)
        self.Yr = np.asarray(self.Yr)
        self.Zr = np.asarray(self.Zr)
        self.Xb = np.asarray(self.Xb)
        self.Yb = np.asarray(self.Yb)
        self.Zb = np.asarray(self.Zb)
        self.U_r = np.asarray(self.U_r)
        self.V_r = np.asarray(self.V_r)
        self.W_r = np.asarray(self.W_r)
        self.U_b = np.asarray(self.U_b)
        self.V_b = np.asarray(self.V_b)
        self.W_b = np.asarray(self.W_b)

    def visualize(self):
        # Visualisation d'une map en 3D
        windows_size = 4.0
        print(f"Visualizing map: {self.map.name} with {len(self.map.notes)} notes")

        # Calcul des bornes pour les fen�tres de temps
        tmin = float(
            min(
                min(self.Yr) if len(self.Yr) else np.inf,
                min(self.Yb) if len(self.Yb) else np.inf,
            )
        )
        tmax = float(
            max(
                max(self.Yr) if len(self.Yr) else -np.inf,
                max(self.Yb) if len(self.Yb) else -np.inf,
            )
        )

        bins = np.arange(tmin, tmax + windows_size, windows_size)

        frames = []
        for start in bins[:-1]:
            end = start + windows_size
            # Filtrer les notes dans la fen�tre de temps
            # mr et mb sont des masques booleens pour les notes rouges et bleues respectivement qui sont dans la fen�tre de temps [start, end)]
            mr = (self.Yr >= start) & (self.Yr < end)
            mb = (self.Yb >= start) & (self.Yb < end)
            frames.append(
                go.Frame(
                    name=f"{start:.2f}-{end:.2f}",
                    data=[
                        go.Scatter3d(
                            x=self.Xr[
                                mr
                            ],  # on garde que les items de Xr, Yr, Zr qui sont dans la fen�tre de temps gr�ce au masque mr
                            y=self.Yr[mr],
                            z=self.Zr[mr],
                            mode="lines+markers",
                            marker=dict(size=3, color="red"),
                            line=dict(color="red", width=3),
                            showlegend=False,
                        ),
                        go.Scatter3d(
                            x=self.Xb[mb],
                            y=self.Yb[mb],
                            z=self.Zb[mb],
                            mode="lines+markers",
                            marker=dict(size=3, color="blue"),
                            line=dict(color="blue", width=3),
                            showlegend=False,
                        ),
                        go.Cone(
                            x=self.Xr[mr],
                            y=self.Yr[mr],
                            z=self.Zr[mr],
                            u=self.U_r[mr],
                            v=self.V_r[mr],
                            w=self.W_r[mr],
                            sizemode="raw",
                            sizeref=0.6,
                            anchor="tail",
                            colorscale=[[0, "red"], [1, "red"]],
                            showscale=False,
                            name="Red Notes",
                            showlegend=False,
                        ),
                        go.Cone(
                            x=self.Xb[mb],
                            y=self.Yb[mb],
                            z=self.Zb[mb],
                            u=self.U_b[mb],
                            v=self.V_b[mb],
                            w=self.W_b[mb],
                            sizemode="raw",
                            sizeref=0.6,
                            anchor="tail",
                            colorscale=[[0, "blue"], [1, "blue"]],
                            showscale=False,
                            name="Blue Notes",
                            showlegend=False,
                        ),
                    ],
                )
            )

        init = frames[0].data if frames else []

        sliders = [
            {
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame.name,
                    }
                    for frame in frames
                ],
                "transition": {"duration": 0},
                "x": 0,
                "y": -0.05,
                "len": 1.0,
            }
        ]

        updatemenus = [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": True}},
                        ],
                    },
                ],
                "x": 0,
                "y": -0.12,
                "direction": "left",
                "showactive": False,
            }
        ]

        fig = go.Figure(
            data=init,
            layout=go.Layout(
                title=f"Map: {self.map.name}",
                scene=dict(
                    xaxis_title="Colonne (0-3)",
                    yaxis_title="Temps (s)",
                    zaxis_title="Etage (0-2)",
                    aspectmode="cube",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                sliders=sliders,
                updatemenus=updatemenus,
            ),
            frames=frames,
        )

        fig.show()
