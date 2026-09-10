# CardDekho Dataset

## Source
This dataset contains used car listings from the Indian automotive marketplace CardDekho.

## Description
- **Records**: 15,411
- **Target Variable**: `selling_price` (INR)
- **Zero null values, zero duplicates**

## Columns

| Column | Type | Description |
|--------|------|-------------|
| car_name | string | Full car name (brand + model) |
| brand | string | Car manufacturer (32 unique) |
| model | string | Car model (120 unique) |
| vehicle_age | integer | Age of vehicle in years |
| km_driven | integer | Kilometers driven |
| seller_type | string | Individual / Dealer / Trustmark Dealer |
| fuel_type | string | Petrol / Diesel / CNG / LPG / Electric |
| transmission_type | string | Manual / Automatic |
| mileage | float | Fuel efficiency in kmpl |
| engine | integer | Engine displacement in cc |
| max_power | float | Maximum power in bhp |
| seats | integer | Number of seats |
| selling_price | integer | Selling price in INR (target) |

## Data Quality Notes
- 2 records have `seats=0` (handled during preprocessing)
- `Isuzu` appears as both "Isuzu" and "ISUZU" (normalized)
- `km_driven` has extreme outliers up to 3.8M (capped at 99th percentile)
- `selling_price` ranges from ₹40K to ₹3.95Cr (right-skewed, capped at 99th percentile)

## Usage
Place the CSV file in this directory. The training pipeline loads it from the path configured in `config.py`.
