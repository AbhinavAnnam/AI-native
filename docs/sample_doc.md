# Software Feature Spec: Payment Processing & Discount Engine

## Module 1: Order Validation & Data Schema
All incoming purchase payloads must be processed through the `OrderPayload` model. 
- `user_id`: String (UUIDv4 required).
- `cart_total`: Float (Must be strictly greater than 0.00).
- `items`: List of dictionaries containing `item_id` and `quantity`.
If `cart_total` is less than or equal to 0, the system must raise an `InvalidOrderAmountException` with HTTP status code 400.

## Module 2: Discount Calculation Logic
The core pricing engine exposes a function `apply_customer_discount(cart_total: float, user_tier: str) -> float`.
Discount rules strictly apply as follows:
- `STANDARD`: 0% discount.
- `GOLD`: 10% discount on total.
- `PLATINUM`: 20% discount on total, plus an additional flat $5 off if `cart_total` exceeds $100.
If an unrecognized `user_tier` string is passed, default to `STANDARD` tier behavior.

## Module 3: Exception Handling & Event Logging
All transaction failures must be caught by custom application exceptions:
- Database connectivity failures raise `PaymentGatewayTimeoutError`.
- Insufficient inventory items raise `StockOutException`.
Every caught error must log a JSON payload to `sys.stderr` containing the timestamp, `user_id`, and exact error traceback before terminating execution.