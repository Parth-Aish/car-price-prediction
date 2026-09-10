"""
Input validation for prediction requests.

Validates types, ranges, and categorical values before data reaches the model.
Returns a list of human-readable error messages.
"""

import logging

logger = logging.getLogger(__name__)

# These will be updated from model metadata at app startup
VALID_BRANDS = []
VALID_MODELS = []
VALID_FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
VALID_TRANSMISSION_TYPES = ["Manual", "Automatic"]
VALID_SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]

CURRENT_YEAR = 2026

REQUIRED_FIELDS = [
    "brand", "model", "year", "kilometers_driven",
    "fuel_type", "transmission", "engine", "mileage", "power", "seats"
]


def init_valid_values(metadata: dict) -> None:
    """Load valid categorical values from model metadata."""
    global VALID_BRANDS, VALID_MODELS
    VALID_BRANDS = metadata.get("valid_brands", [])
    VALID_MODELS = metadata.get("valid_models", [])
    logger.info(
        "Validation initialized: %d brands, %d models",
        len(VALID_BRANDS), len(VALID_MODELS)
    )


def validate_prediction_input(data: dict) -> list[str]:
    """
    Validate a prediction request payload.
    Returns an empty list if valid, otherwise a list of error messages.
    """
    errors = []

    if not data or not isinstance(data, dict):
        return ["Request body must be a JSON object"]

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data or data[field] is None or data[field] == "":
            errors.append(f"'{field}' is required")

    if errors:
        return errors  # Stop early — can't validate values of missing fields

    # Type and range checks
    # Brand
    brand = data.get("brand")
    if not isinstance(brand, str):
        errors.append("'brand' must be a string")
    elif VALID_BRANDS and brand not in VALID_BRANDS:
        errors.append(f"Unknown brand '{brand}'. Valid brands: {', '.join(VALID_BRANDS[:10])}...")

    # Model
    model = data.get("model")
    if not isinstance(model, str):
        errors.append("'model' must be a string")

    # Year
    year = data.get("year")
    try:
        year = int(year)
        if year < 1990:
            errors.append(f"'year' must be 1990 or later, got {year}")
        elif year > CURRENT_YEAR:
            errors.append(f"'year' cannot be in the future (max: {CURRENT_YEAR})")
    except (TypeError, ValueError):
        errors.append("'year' must be an integer")

    # Kilometers driven
    km = data.get("kilometers_driven")
    try:
        km = int(km)
        if km < 0:
            errors.append("'kilometers_driven' cannot be negative")
        elif km > 1_000_000:
            errors.append("'kilometers_driven' seems unrealistic (max: 1,000,000)")
    except (TypeError, ValueError):
        errors.append("'kilometers_driven' must be a number")

    # Fuel type
    fuel = data.get("fuel_type")
    if not isinstance(fuel, str):
        errors.append("'fuel_type' must be a string")
    elif fuel not in VALID_FUEL_TYPES:
        errors.append(f"Invalid fuel_type '{fuel}'. Options: {', '.join(VALID_FUEL_TYPES)}")

    # Transmission
    trans = data.get("transmission")
    if not isinstance(trans, str):
        errors.append("'transmission' must be a string")
    elif trans not in VALID_TRANSMISSION_TYPES:
        errors.append(
            f"Invalid transmission '{trans}'. Options: {', '.join(VALID_TRANSMISSION_TYPES)}"
        )

    # Engine (cc)
    engine = data.get("engine")
    try:
        engine = float(engine)
        if engine <= 0:
            errors.append("'engine' must be positive")
        elif engine > 10000:
            errors.append("'engine' seems unrealistic (max: 10000 cc)")
    except (TypeError, ValueError):
        errors.append("'engine' must be a number")

    # Mileage (kmpl)
    mileage = data.get("mileage")
    try:
        mileage = float(mileage)
        if mileage < 0:
            errors.append("'mileage' cannot be negative")
        elif mileage > 100:
            errors.append("'mileage' seems unrealistic (max: 100 kmpl)")
    except (TypeError, ValueError):
        errors.append("'mileage' must be a number")

    # Power (bhp)
    power = data.get("power")
    try:
        power = float(power)
        if power <= 0:
            errors.append("'power' must be positive")
        elif power > 1500:
            errors.append("'power' seems unrealistic (max: 1500 bhp)")
    except (TypeError, ValueError):
        errors.append("'power' must be a number")

    # Seats
    seats = data.get("seats")
    try:
        seats = int(seats)
        if seats < 2 or seats > 10:
            errors.append(f"'seats' must be between 2 and 10, got {seats}")
    except (TypeError, ValueError):
        errors.append("'seats' must be an integer")

    return errors
