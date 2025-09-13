#!/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import column, gridplot, layout
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import ColumnDataSource, figure
from sklearn.metrics import confusion_matrix


def visualizze_paths(
    gt_path, pred_path, html_file="", title="VO exercises", file_out="plot.html"
):
    output_file(file_out, title=html_file)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"
    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)
    source = ColumnDataSource(
        data=dict(
            gtx=gt_path[:, 0],
            gty=gt_path[:, 1],
            px=pred_path[:, 0],
            py=pred_path[:, 1],
            diffx=np.arange(len(diff)),
            diffy=diff,
            disx=xs,
            disy=ys,
        )
    )

    fig1 = figure(
        title="Paths",
        tools=tools,
        match_aspect=True,
        width_policy="max",
        toolbar_location="above",
        x_axis_label="x",
        y_axis_label="y",
    )
    fig1.circle("gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT")
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle("px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed")
    fig1.legend.click_policy = "hide"

    fig2 = figure(
        title="Error",
        tools=tools,
        width_policy="max",
        toolbar_location="above",
        x_axis_labels="frame",
        y_axis_label="error"
    )
    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(layout(
        [
            Div(text=f"<h1>{title}</h1>"),
            Div(text="<h2>Paths</h2>"),
            [fig1, fig2],
         ], sizing_mode='scale_width')
    )


def make_residual_plot(x, residual_init, residual_minimized):


