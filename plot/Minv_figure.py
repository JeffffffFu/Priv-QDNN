import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def Minv_SSIM_MNIST():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2]*6
        ),
        'Value': [
            # Pytorch-quan
            0.461, 0.469, 0.478, 0.485, 0.836,
            # AdaRound
            0.462, 0.469, 0.471, 0.513, 0.892,
            # QDrop
            0.557, 0.643, 0.656, 0.667, 0.889,
            # PACT
            0.544, 0.613, 0.617, 0.865, 0.875,
            # HAWQ
            0.448, 0.459, 0.463, 0.468, 0.476,
            # BSQ
            0.522, 0.524, 0.551, 0.578, 0.589
        ]
    }
    baseline = 0.462  # Original 32-bit full-precision
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(8, 6))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['Value'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Full-precision'
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('Avg of SSIM', fontsize=22)
    plt.ylim(bottom=0.44, top=0.9)  # Adjust the y-axis range to fit the SSIM data
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
   # plt.legend(loc='upper left', fontsize=22)
    plt.tight_layout()
    plt.show()


def Minv_SSIM_FMNIST():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2]*6
        ),
        'Value': [
            # Pytorch-quan
            0.481, 0.491, 0.499, 0.501, 0.783,
            # AdaRound
            0.481, 0.483, 0.486, 0.489, 0.85,
            # QDrop
            0.556, 0.612, 0.631, 0.648, 0.874,
            # PACT
            0.543, 0.55, 0.561, 0.848, 0.875,
            # HAWQ
            0.478, 0.478, 0.479, 0.481, 0.682,
            # BSQ
            0.493,0.498, 0.503, 0.509, 0.512
        ]
    }
    baseline = 0.479  # Original 32-bit full-precision
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(8, 6))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['Value'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Full-precision'
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('Avg of SSIM', fontsize=22)
    plt.ylim(bottom=0.46, top=0.88)  # Adjust the y-axis range to fit the FMNIST SSIM data
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
  #  plt.legend(loc='upper left', fontsize=22)
    plt.tight_layout()
    plt.show()


def Minv_Memorization_Capacity_MNIST():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
                ['Pytorch-quan'] * 5 + ['AdaRound'] * 5 + ['QDrop'] * 5 + ['PACT'] * 5 + ['HAWQ'] * 5 + ['BSQ'] * 5
        ),
        'Bit': (
                [10, 8, 6, 4, 2] * 6
        ),
        'Value': [
            # Pytorch-quan
            46872.50, 46489.00, 46320.00, 45665.00, 7085.00,
            # AdaRound
            37785.00, 37040.00, 37460.00, 36945.00, 5070.00,
            # QDrop
            26055.00, 25805.00, 25040.00, 24740.00, 5045.00,
            # PACT
            43035.00, 43025.00, 42485.00, 10225.00, 5045.00,
            # HAWQ
            46325.00, 45910.00, 45525.00, 45045.00, 42635.00,
            # BSQ
            43085.00, 42645.00, 42535.00, 42145.00, 40060.00
        ]
    }
    baseline = 46183.63  # Original 32-bit full-precision
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['Value'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Full-precision'
    )
    plt.xlabel('Bits (W/A)', fontsize=28)
    plt.ylabel('Memorization capacity', fontsize=28)
    plt.ylim(bottom=0, top=50000)
    plt.xticks(bit_order, bit_labels, fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='lower left', fontsize=28)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Minv_SSIM_MNIST()
    # Minv_SSIM_FMNIST()
    Minv_Memorization_Capacity_MNIST()
