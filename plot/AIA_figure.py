import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def AIA_Pearson_Correlation_UTKFace():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*4 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
        ),
        'Pearson': [
            -0.2079, -0.2029, -0.2017, -0.1871, -0.118, # Pytorch-quan
            -0.2261, -0.2154, -0.1747, -0.1766,         # AdaRound
            -0.2143, -0.2097, -0.1752, -0.1742, -0.012,  # QDrop
            -0.172, -0.171, -0.1624, -0.1504, -0.0896,   # PACT
            -0.199, -0.1957, -0.1889, -0.1854, -0.1834,  # HAWQ
            -0.1797, -0.1776, -0.1743, -0.1741, -0.1782  # BSQ
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    df['Pearson'] = df['Pearson'].abs()
    original_pearson = 0.224
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        if model == 'AdaRound':
            mask = (df['Model'] == model) & (df['Bit'] != '2')
        else:
            mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['Pearson'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_pearson,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('Pearson correlation (abs)', fontsize=22)
    plt.ylim(0.0, 0.23)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

def AIA_Pearson_Correlation_CelebA():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2]*6
        ),
        'Pearson': [
            # Pytorch-quan
            0.2943, 0.2931, 0.2912, 0.287, 0.2,
            # AdaRound
            0.295, 0.2851, 0.285, 0.2825, 0.1974,
            # QDrop
            0.2841, 0.2797, 0.2716, 0.2613, 0.015,
            # PACT
            0.2431, 0.2433, 0.2411, 0.2379, 0.193,
            # HAWQ
            0.2899, 0.2898, 0.2836, 0.2836, 0.2814,
            # BSQ
            0.2892, 0.2868, 0.2834, 0.2828, 0.2803
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    original_pearson = 0.2916  # Original 32-bit full-precision
    plt.figure(figsize=(10, 7))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['Pearson'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_pearson,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=30)
    plt.ylabel('Pearson correlation (abs)', fontsize=30)
    plt.ylim(0.0, 0.3)  # Adjust the y-axis range to fit the new data
    plt.xticks(bit_order, bit_labels, fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.show()
# === UTKFace white-box, single figure ===
def plot_aia_utkface_whitebox():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*4 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
        ),
        'WhiteBox': [
            52.81, 54.27, 52.75, 52.77, 38.22, # Pytorch-quan
            52.74, 53.68, 52.29, 50.11,        # AdaRound
            53.28, 53.01, 51.43, 47.88, 25.09, # QDrop
            50.97, 50.18, 50.77, 42.73, 36.19, # PACT
            52.77, 53.00, 52.26, 52.97, 52.69, # HAWQ
            53.10, 53.35, 53.66, 53.35, 52.75  # BSQ
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    original_white = 52.34
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        if model == 'AdaRound':
            mask = (df['Model'] == model) & (df['Bit'] != '2')
        else:
            mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['WhiteBox'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_white,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=28)
    plt.ylabel('Attack Accuracy (%)', fontsize=28)
    plt.ylim(25, 55)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_aia_utkface_whitebox():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] * 6
        ),
        'WhiteBox': [
            54.23, 54.09, 53.62, 52.70, 38.50, # Pytorch-quan
            53.91, 53.88, 53.44, 52.12, 37.55, # AdaRound
            54.14, 53.09, 52.62, 46.88, 25.09, # QDrop
            50.14, 50.11, 50.55, 40.17, 32.25, # PACT
            53.61, 53.32, 53.21, 52.71, 52.45, # HAWQ
            53.35, 52.95, 52.96, 52.46, 51.98  # BSQ
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    original_white = 54.03
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['WhiteBox'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_white,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=28)
    plt.ylabel('Attack Accuracy (%)', fontsize=28)
    plt.ylim(25, 55)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_aia_utkface_blackbox():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] * 6
        ),
        'BlackBox': [
            39.48, 38.38, 36.31, 34.23, 33.07, # Pytorch-quan
            38.23, 37.14, 36.88, 35.86, 31.61, # AdaRound
            38.76, 38.61, 37.88, 33.93, 25.09, # QDrop
            29.97, 28.22, 27.63, 27.34, 27.04, # PACT
            35.37, 37.06, 36.55, 33.81, 35.14, # HAWQ
            33.80, 34.73, 33.46, 34.29, 32.19  # BSQ
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    original_black = 36.38
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['BlackBox'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_black,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('Attack Accuracy (%)', fontsize=22)
    plt.ylim(25, 40)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

# === CelebA black-box, single figure ===
def plot_aia_celeba_blackbox():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] * 6
        ),
        'BlackBox': [
            44.25, 44.04, 43.98, 43.11, 40.71, # Pytorch-quan
            44.47, 44.25, 44.14, 44.16, 40.19, # AdaRound
            44.32, 44.22, 44.33, 44.16, 24.59, # QDrop
            41.85, 42.20, 41.47, 40.19, 37.18, # PACT
            44.31, 44.15, 44.05, 44.05, 44.02, # HAWQ
            45.00, 45.06, 44.86, 44.58, 44.26  # BSQ
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    original_black = 44.39
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['BlackBox'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_black,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('Attack Accuracy (%)', fontsize=22)
    plt.ylim(23, 46)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

def plot_aia_celeba_whitebox():
    import matplotlib.pyplot as plt
    import pandas as pd
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    model_order = ['Pytorch-quan', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    data = {
        'Model': (
            ['Pytorch-quan']*5 + ['AdaRound']*5 + ['QDrop']*5 + ['PACT']*5 + ['HAWQ']*5 + ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] * 6
        ),
        'WhiteBox': [
            71.79, 71.79, 70.52, 66.21, 46.11, # Pytorch-quan
            71.97, 70.92, 70.62, 67.57, 46.03, # AdaRound
            71.17, 70.73, 66.97, 59.44, 25.68, # QDrop
            54.22, 51.58, 51.38, 51.59, 47.36, # PACT
            68.98, 68.91, 68.00, 65.68, 63.73, # HAWQ
            71.27, 70.98, 70.92, 70.08, 66.36  # BSQ
        ]
    }
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    original_white = 71.84
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        mask = (df['Model'] == model)
        subdf = df[mask]
        plt.plot(
            subdf['Bit'],
            subdf['WhiteBox'],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10
        )
    plt.axhline(
        y=original_white,
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('Attack Accuracy (%)', fontsize=22)
    plt.ylim(25, 75)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

# === AIA UTKFace Pytorch-quan relative change rate, single figure ===
def plot_aia_celeba_relative_pytorchquan():
    data = get_aia_data()['CelebA']['Pytorch-quan']
    baseline = get_aia_data()['CelebA']['baseline']
    bit_order = data['bit']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    rel_white = [abs(w - baseline['white']) for w in data['white']]
    rel_black = [abs(b - baseline['black']) for b in data['black']]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, rel_white, color='#E41A1C', marker='o', label='White-box AIA')
    plt.plot(bit_order, rel_black, color='#377EB8', marker='s', label='Black-box AIA')
    plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Accuracy Change (%)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# === AIA UTKFace AdaRound relative change rate, single figure ===
def plot_aia_celeba_relative_adaround():
    data = get_aia_data()['CelebA']['AdaRound']
    baseline = get_aia_data()['CelebA']['baseline']
    bit_order = data['bit']
    bit_labels = ['10/10', '8/8', '6/6', '4/4']
    rel_white = [abs(w - baseline['white']) for w in data['white']]
    rel_black = [abs(b - baseline['black']) for b in data['black']]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, rel_white, color='#E41A1C', marker='o', label='White-box AIA')
    plt.plot(bit_order, rel_black, color='#377EB8', marker='s', label='Black-box AIA')
    plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Accuracy Change (%)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# === AIA UTKFace QDrop relative change rate, single figure ===
def plot_aia_celeba_relative_qdrop():
    data = get_aia_data()['CelebA']['QDrop']
    baseline = get_aia_data()['CelebA']['baseline']
    bit_order = data['bit']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    rel_white = [abs(w - baseline['white']) for w in data['white']]
    rel_black = [abs(b - baseline['black']) for b in data['black']]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, rel_white, color='#E41A1C', marker='o', label='White-box AIA')
    plt.plot(bit_order, rel_black, color='#377EB8', marker='s', label='Black-box AIA')
    plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Accuracy Change (%)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# === AIA UTKFace PACT relative change rate, single figure ===
def plot_aia_celeba_relative_pact():
    data = get_aia_data()['CelebA']['PACT']
    baseline = get_aia_data()['CelebA']['baseline']
    bit_order = data['bit']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    rel_white = [abs(w - baseline['white']) for w in data['white']]
    rel_black = [abs(b - baseline['black']) for b in data['black']]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, rel_white, color='#E41A1C', marker='o', label='White-box AIA')
    plt.plot(bit_order, rel_black, color='#377EB8', marker='s', label='Black-box AIA')
    plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Accuracy Change (%)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

# === AIA UTKFace HAWQ relative change rate, single figure ===
def plot_aia_celeba_relative_hawq():
    data = get_aia_data()['CelebA']['HAWQ']
    baseline = get_aia_data()['CelebA']['baseline']
    bit_order = data['bit']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    rel_white = [abs(w - baseline['white']) for w in data['white']]
    rel_black = [abs(b - baseline['black']) for b in data['black']]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, rel_white, color='#E41A1C', marker='o', label='White-box AIA')
    plt.plot(bit_order, rel_black, color='#377EB8', marker='s', label='Black-box AIA')
    plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Accuracy Change (%)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_aia_celeba_relative_bsq():
    data = get_aia_data()['CelebA']['BSQ']
    baseline = get_aia_data()['CelebA']['baseline']
    bit_order = data['bit']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    rel_white = [abs(w - baseline['white']) for w in data['white']]
    rel_black = [abs(b - baseline['black']) for b in data['black']]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, rel_white, color='#E41A1C', marker='o', label='White-box AIA')
    plt.plot(bit_order, rel_black, color='#377EB8', marker='s', label='Black-box AIA')
    plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Accuracy Change (%)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

def get_aia_data():
    return {
        'UTKFace': {
            'baseline': {'white': 54.03, 'black': 36.38},
            'Pytorch-quan': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [54.23, 54.09, 53.62, 52.70, 38.50],
                'black': [39.48, 38.38, 36.31, 34.23, 33.07]
            },
            'AdaRound': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [53.91, 53.88, 53.44, 52.12, 37.55],
                'black': [38.23, 37.14, 36.88, 35.86, 31.61]
            },
            'QDrop': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [54.14, 53.09, 52.62, 46.88, 25.09],
                'black': [38.76, 38.61, 37.88, 33.93, 25.09]
            },
            'PACT': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [50.14, 50.11, 50.55, 40.17, 32.25],
                'black': [29.97, 28.22, 27.63, 27.34, 27.04]
            },
            'HAWQ': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [53.61, 53.32, 53.21, 52.71, 52.45],
                'black': [35.37, 37.06, 36.55, 33.81, 35.14]
            },
            'BSQ': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [53.35, 52.95, 52.96, 52.46, 51.98],
                'black': [33.80, 34.73, 33.46, 34.29, 32.19]
            }
        },
        'CelebA': {
            'baseline': {'white': 71.84, 'black': 44.39},
            'Pytorch-quan': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [71.79, 71.79, 70.52, 66.21, 46.11],
                'black': [44.25, 44.04, 43.98, 43.11, 40.71]
            },
            'AdaRound': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [71.97, 70.92, 70.62, 67.57, 46.03],
                'black': [44.47, 44.25, 44.14, 44.16, 40.19]
            },
            'QDrop': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [71.17, 70.73, 66.97, 59.44, 25.68],
                'black': [44.32, 44.22, 44.33, 44.16, 24.59]
            },
            'PACT': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [54.22, 51.58, 51.38, 51.59, 47.36],
                'black': [41.85, 42.20, 41.47, 40.19, 37.18]
            },
            'HAWQ': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [68.98, 68.91, 68.00, 65.68, 63.73],
                'black': [44.31, 44.15, 44.05, 44.05, 44.02]
            },
            'BSQ': {
                'bit': ['10', '8', '6', '4', '2'],
                'white': [71.27, 70.98, 70.92, 70.08, 66.36],
                'black': [45.00, 45.06, 44.86, 44.58, 44.26]
            }
        }
    }





def plot_aia_distance_bsq():
    bit_order = ['10', '8', '6', '4', '2']
    bit_labels = ['10/10', '8/8', '6/6', '4/4', '2/2']
    posterior_probabilities = [0.055, 0.054, 0.051, 0.054, 0.056]
    last_layer_output = [0.516, 0.614, 0.622, 0.549, 0.604]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(bit_order, posterior_probabilities, color='#E41A1C', marker='o', label='Posterior Probabilities', linewidth=2.5, markersize=10)
    plt.plot(bit_order, last_layer_output, color='#377EB8', marker='s', label='Last Layer Output', linewidth=2.5, markersize=10)
  #  plt.axhline(y=0, color='gray', linestyle=(0, (3, 8)), linewidth=3.5, alpha=0.7)
    plt.xlabel('Bits (W/A)', fontsize=20)
    plt.ylabel('Distance (L2 norm)', fontsize=22)
    plt.xticks(bit_order, bit_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
   # AIA_Pearson_Correlation_UTKFace()
   AIA_Pearson_Correlation_CelebA()

   #plot_aia_utkface_whitebox()
   #plot_aia_utkface_blackbox()
  #  plot_aia_celeba_whitebox()
   # plot_aia_celeba_blackbox()
  #  plot_aia_celeba_relative_bsq()
   # plot_aia_celeba_relative_adaround()
  #  plot_aia_celeba_relative_hawq()
   # plot_aia_celeba_relative_qdrop()
   # plot_aia_celeba_relative_pytorchquan()
  #  plot_aia_celeba_relative_pact()
   #plot_aia_distance_bsq()

