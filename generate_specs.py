import pandas as pd
import json

df = pd.read_csv('data/cardekho_dataset.csv')

agg_num = df.groupby(['brand', 'model'])[['engine', 'mileage', 'max_power', 'seats']].median().reset_index()
agg_cat_fuel = df.groupby(['brand', 'model'])['fuel_type'].agg(lambda x: x.mode()[0]).reset_index()
agg_cat_trans = df.groupby(['brand', 'model'])['transmission_type'].agg(lambda x: x.mode()[0]).reset_index()

out = {}
for _, row in agg_num.iterrows():
    b, m = row['brand'], row['model']
    if b not in out:
        out[b] = {}
    out[b][m] = {
        'engine': int(row['engine']),
        'mileage': float(row['mileage']),
        'power': float(row['max_power']),
        'seats': int(row['seats'])
    }

for _, row in agg_cat_fuel.iterrows():
    out[row['brand']][row['model']]['fuel_type'] = row['fuel_type']

for _, row in agg_cat_trans.iterrows():
    out[row['brand']][row['model']]['transmission'] = row['transmission_type']

with open('data/car_specs.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"Saved specs for {sum(len(m) for m in out.values())} models")
