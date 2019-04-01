"""Lookup data for plotting styles"""

model_style = {
    'B3LYP': {'linestyle': ':', 'color': 'k', 'marker': 'o', 'label': 'B3LYP'},
    'g4mp2': {'linestyle': '-', 'color': 'mediumblue', 'marker': 'o', 'label': 'SchNet'},
    'g4mp2-charges': {'linestyle': '-', 'color': 'orange', 'marker': 's',
                      'label': 'SchNet Charges'},
    'g4mp2-charges-in-outnet': {'linestyle': '-', 'color': 'forestgreen',
                                'marker': '*', 'label': 'SchNet Charges in Outnet'},
    'g4mp2-delta': {'linestyle': '-', 'color': 'crimson', 'marker': '^', 'label': 'SchNet Delta',
                   'markerfacecolor': 'none'},
    'g4mp2-multitask': {'linestyle': '-', 'color': 'purple',
                        'marker': 'v', 'label': 'SchNet Multitask'},
    'g4mp2-stacked-delta': {'linestyle': '-', 'color': 'sienna',
                            'marker': 'x', 'label': 'SchNet Stacked'},
    'g4mp2-transfer': {'linestyle': '-', 'color': 'lightseagreen',
                       'marker': '.', 'label': 'SchNet Transfer'},
    'FCHL': {'linestyle': '--', 'color': 'gray', 'marker': 'o', 'label': 'FCHL'},
    'FCHL Delta': {'linestyle': '--', 'color': 'goldenrod',
                   'marker': 's', 'label': 'FCHL Delta', 'markerfacecolor': 'none'}
}
