"""
from: https://community.plotly.com/t/help-with-3d-scatter-plot-animation-slider-doesnt-update/38107
"""

#!/usr/bin/python3

import pandas as pd
import plotly.graph_objects as go
from io import StringIO

NUM_FRAMES = 4
TESTDATA = StringIO("""frame_id,object_id,joint_id,x,y,z
0,0,0,23.972859220560373,22.122234181663273,5.133874974568155
0,0,1,24.755578569389865,22.287544285896196,5.142875625876776
0,0,2,26.26817985399368,23.675066255203753,5.017036418972257
0,1,0,25.24414502991678,20.825589182023478,5.511625155662896
0,1,1,24.511099773907866,20.764126911431433,5.398725514026496
0,1,2,22.497134678608056,21.360955588373763,5.342810998523781
1,0,0,23.810306017845463,22.22337497470969,5.101599296213683
1,0,1,24.545499103404087,22.28504362225052,5.1238557962297655
1,0,2,26.222459728645394,23.519275132333277,5.039497549239671
1,1,0,25.344313713951284,20.929643010445705,5.50330414247449
1,1,1,24.663587071887555,20.86881736802757,5.3951547661260735
1,1,2,22.50504995405339,21.20725201551635,5.369886028029748
2,0,0,23.75703530317526,22.274973017146227,5.143702258466614
2,0,1,24.439338516075107,22.336030840885144,5.112123095992956
2,0,2,26.277726601807373,23.256682889164118,5.09109647379112
2,1,0,25.397459549159425,21.03549166549144,5.420544301008246
2,1,1,24.71869744147468,20.87094465901064,5.4135730040028855
2,1,2,22.503075969335587,21.048072308781315,5.305362866779313
3,0,0,23.651038782172837,22.272996860776033,5.1764506996884245
3,0,1,24.385851986554748,22.38757577597525,5.1539425790066105
3,0,2,26.38603162292819,22.995161570973796,5.09931670535051
3,1,0,25.4983870991434,21.087903810058723,5.511071537968743
3,1,1,24.76744309472927,20.922507200181972,5.4511983917338656
3,1,2,22.51022643773005,20.89393185376788,5.282897556380078
        """)


def make_scatter_one_frame(df, frame_id):
    df_cur_frame = df.loc[df['frame_id'] == frame_id]
    frame = []

    colors = ["#440154", "#b5de2b"]

    for i, object_id in enumerate(df_cur_frame['object_id'].unique()):
        df_cur_object = df_cur_frame[df_cur_frame['object_id'] == object_id]

        scatter3d = go.Scatter3d(
            x=df_cur_object['x'], y=df_cur_object['y'], z=df_cur_object['z'],
            marker=dict(
                size=4,
                color=colors[i]
            ),
            mode='lines+markers',
            line=dict(
                color=colors[i],
                width=2
            )
        )

        frame.append(scatter3d)
    return frame


def make_frames(df):
   frames = []

   for frame_id in range(NUM_FRAMES):

       cur_frame_plots = make_scatter_one_frame(df, frame_id)

       frames.append(go.Frame(data=cur_frame_plots,
                              name='{}'.format(frame_id),  #I ADDED THIS LINE OF CODE!!!!!!
                              layout=go.Layout(title="{}".format(frame_id))))
   return frames


if __name__ == "__main__":

    df = pd.read_csv(TESTDATA)

    steps = [dict(method='animate',
                  args=[["{}".format(k)],
                        dict(mode='immediate',
                             frame=dict(duration=300),
                             transition=dict(duration=0)
                             )
                        ],
                  label="{}".format(k)
                  ) for k in range(NUM_FRAMES)]

    sliders = [
        dict(
            x=0.1,
            y=0,
            len=0.9,
            pad=dict(b=10, t=50),
            active=0,
            steps=steps,
            currentvalue=dict(font=dict(size=20), prefix="", visible=True, xanchor='right'),
            transition=dict(easing="cubic-in-out", duration=300))
    ]

    fig = go.Figure(data=make_scatter_one_frame(df, 0),
                    layout=go.Layout(scene=dict(xaxis=dict(range=[0, 40],
                                                           tickmode='linear', tick0=0, dtick=5),
                                                yaxis=dict(range=[0, 40],
                                                           tickmode='linear', tick0=0, dtick=5),
                                                zaxis=dict(range=[0, 40],
                                                           tickmode='linear', tick0=0, dtick=5)),
                                     sliders=sliders,
                                     updatemenus=[dict(
                                         type="buttons",
                                         buttons=[dict(label="Play",
                                                       method="animate",
                                                       args=[None, dict(frame=dict(duration=100))])])],
                                     ),
                    frames=make_frames(df))

    fig.show()


