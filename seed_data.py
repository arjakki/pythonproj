"""
Seed MySQL with customers, products, orders, and order_items (grocery history).
Run: python seed_data.py
"""
import os, random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

def get_conn():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        autocommit=False,
    )

# ── Schema ────────────────────────────────────────────────────────────────────
DDL = [
    """CREATE TABLE IF NOT EXISTS customers (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        name       VARCHAR(100) NOT NULL,
        email      VARCHAR(150) UNIQUE NOT NULL,
        phone      VARCHAR(20),
        city       VARCHAR(80),
        joined_at  DATE NOT NULL,
        INDEX idx_city (city)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS products (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        name       VARCHAR(150) NOT NULL,
        category   VARCHAR(60)  NOT NULL,
        price      DECIMAL(8,2) NOT NULL,
        unit       VARCHAR(20)  NOT NULL,
        in_stock   TINYINT(1)   DEFAULT 1,
        INDEX idx_cat (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS orders (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        customer_id INT NOT NULL,
        status      ENUM('pending','processing','delivered','cancelled') DEFAULT 'delivered',
        total       DECIMAL(10,2) NOT NULL,
        ordered_at  DATETIME NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        INDEX idx_cust (customer_id),
        INDEX idx_date (ordered_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS order_items (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        order_id   INT NOT NULL,
        product_id INT NOT NULL,
        quantity   INT NOT NULL,
        unit_price DECIMAL(8,2) NOT NULL,
        FOREIGN KEY (order_id)   REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id),
        INDEX idx_order (order_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]

# ── Data ──────────────────────────────────────────────────────────────────────
CUSTOMERS = [
    ("Alice Johnson",  "alice@email.com",   "555-0101", "New York"),
    ("Bob Martinez",   "bob@email.com",     "555-0102", "Los Angeles"),
    ("Carol White",    "carol@email.com",   "555-0103", "Chicago"),
    ("David Lee",      "david@email.com",   "555-0104", "Houston"),
    ("Emma Davis",     "emma@email.com",    "555-0105", "Phoenix"),
    ("Frank Wilson",   "frank@email.com",   "555-0106", "Philadelphia"),
    ("Grace Kim",      "grace@email.com",   "555-0107", "San Antonio"),
    ("Henry Brown",    "henry@email.com",   "555-0108", "San Diego"),
    ("Isla Taylor",    "isla@email.com",    "555-0109", "Dallas"),
    ("Jack Anderson",  "jack@email.com",    "555-0110", "San Jose"),
    ("Karen Thomas",   "karen@email.com",   "555-0111", "Austin"),
    ("Liam Jackson",   "liam@email.com",    "555-0112", "Jacksonville"),
    ("Mia Harris",     "mia@email.com",     "555-0113", "Fort Worth"),
    ("Noah Martin",    "noah@email.com",    "555-0114", "Columbus"),
    ("Olivia Garcia",  "olivia@email.com",  "555-0115", "Charlotte"),
    ("Paul Rodriguez", "paul@email.com",    "555-0116", "Indianapolis"),
    ("Quinn Lewis",    "quinn@email.com",   "555-0117", "San Francisco"),
    ("Rachel Walker",  "rachel@email.com",  "555-0118", "Seattle"),
    ("Sam Hall",       "sam@email.com",     "555-0119", "Denver"),
    ("Tina Allen",     "tina@email.com",    "555-0120", "Nashville"),
]

PRODUCTS = [
    # Fruits
    ("Bananas",            "Fruits",      0.59, "bunch"),
    ("Gala Apples",        "Fruits",      1.29, "lb"),
    ("Strawberries",       "Fruits",      3.99, "pint"),
    ("Blueberries",        "Fruits",      4.49, "pint"),
    ("Navel Oranges",      "Fruits",      0.89, "each"),
    ("Watermelon",         "Fruits",      5.99, "each"),
    ("Grapes (Red)",       "Fruits",      2.99, "lb"),
    ("Avocados",           "Fruits",      1.49, "each"),
    # Vegetables
    ("Broccoli",           "Vegetables",  1.79, "head"),
    ("Spinach",            "Vegetables",  2.49, "bag"),
    ("Carrots",            "Vegetables",  1.29, "lb"),
    ("Roma Tomatoes",      "Vegetables",  0.99, "lb"),
    ("Bell Peppers",       "Vegetables",  0.79, "each"),
    ("Zucchini",           "Vegetables",  0.99, "each"),
    ("Sweet Potatoes",     "Vegetables",  1.09, "lb"),
    ("Garlic",             "Vegetables",  0.49, "bulb"),
    # Dairy
    ("Whole Milk",         "Dairy",       3.49, "gallon"),
    ("Cheddar Cheese",     "Dairy",       4.99, "block"),
    ("Greek Yogurt",       "Dairy",       1.79, "cup"),
    ("Butter",             "Dairy",       3.99, "lb"),
    ("Eggs (Dozen)",       "Dairy",       3.29, "dozen"),
    ("Mozzarella",         "Dairy",       3.49, "ball"),
    # Bakery
    ("Sourdough Bread",    "Bakery",      3.99, "loaf"),
    ("Whole Wheat Bread",  "Bakery",      2.99, "loaf"),
    ("Croissants",         "Bakery",      5.49, "pack"),
    ("Bagels",             "Bakery",      3.49, "pack"),
    # Meat
    ("Chicken Breast",     "Meat",        5.99, "lb"),
    ("Ground Beef",        "Meat",        6.49, "lb"),
    ("Salmon Fillet",      "Meat",        9.99, "lb"),
    ("Bacon",              "Meat",        5.49, "pack"),
    ("Pork Chops",         "Meat",        4.99, "lb"),
    # Pantry
    ("Olive Oil",          "Pantry",      7.99, "bottle"),
    ("Pasta",              "Pantry",      1.49, "box"),
    ("Rice (Basmati)",     "Pantry",      3.99, "bag"),
    ("Canned Tomatoes",    "Pantry",      1.29, "can"),
    ("Peanut Butter",      "Pantry",      3.49, "jar"),
    ("Honey",              "Pantry",      5.99, "jar"),
    ("Oats",               "Pantry",      3.29, "bag"),
    ("Cereal",             "Pantry",      4.49, "box"),
    # Beverages
    ("Orange Juice",       "Beverages",   3.99, "carton"),
    ("Sparkling Water",    "Beverages",   5.49, "pack"),
    ("Almond Milk",        "Beverages",   3.79, "carton"),
    ("Coffee Beans",       "Beverages",  10.99, "bag"),
    ("Green Tea",          "Beverages",   4.99, "box"),
    # Frozen
    ("Frozen Pizza",       "Frozen",      6.99, "each"),
    ("Ice Cream",          "Frozen",      4.49, "pint"),
    ("Frozen Broccoli",    "Frozen",      2.29, "bag"),
    ("Frozen Berries",     "Frozen",      4.99, "bag"),
    # Snacks
    ("Tortilla Chips",     "Snacks",      3.99, "bag"),
    ("Dark Chocolate",     "Snacks",      2.99, "bar"),
]

STATUSES = ["delivered"] * 8 + ["processing"] * 1 + ["cancelled"] * 1

def rand_date(days_back=365):
    return datetime.now() - timedelta(days=random.randint(0, days_back),
                                      hours=random.randint(0, 23),
                                      minutes=random.randint(0, 59))

def main():
    conn = get_conn()
    cur  = conn.cursor()

    print("Creating tables…")
    for ddl in DDL:
        cur.execute(ddl)
    conn.commit()

    # wipe existing seed data
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    for tbl in ("order_items", "orders", "products", "customers"):
        cur.execute(f"TRUNCATE TABLE `{tbl}`")
    cur.execute("SET FOREIGN_KEY_CHECKS=1")
    conn.commit()

    # ── customers ────────────────────────────────────────────
    print("Inserting customers…")
    for name, email, phone, city in CUSTOMERS:
        joined = rand_date(730).date()
        cur.execute(
            "INSERT INTO customers (name,email,phone,city,joined_at) VALUES (%s,%s,%s,%s,%s)",
            (name, email, phone, city, joined),
        )
    conn.commit()

    # ── products ─────────────────────────────────────────────
    print("Inserting products…")
    for name, cat, price, unit in PRODUCTS:
        cur.execute(
            "INSERT INTO products (name,category,price,unit) VALUES (%s,%s,%s,%s)",
            (name, cat, price, unit),
        )
    conn.commit()

    cur.execute("SELECT id FROM customers")
    cust_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT id, price FROM products")
    prod_rows = cur.fetchall()   # [(id, price), …]

    # ── orders + order_items ─────────────────────────────────
    print("Inserting orders and items…")
    order_count = 0
    for _ in range(180):          # 180 orders spread across customers
        cust_id    = random.choice(cust_ids)
        ordered_at = rand_date(365)
        status     = random.choice(STATUSES)
        n_items    = random.randint(2, 8)
        items      = random.sample(prod_rows, n_items)

        total = 0.0
        item_rows = []
        for pid, pprice in items:
            qty = random.randint(1, 4)
            line = float(pprice) * qty
            total += line
            item_rows.append((pid, qty, float(pprice)))

        cur.execute(
            "INSERT INTO orders (customer_id,status,total,ordered_at) VALUES (%s,%s,%s,%s)",
            (cust_id, status, round(total, 2), ordered_at),
        )
        order_id = cur.lastrowid
        for pid, qty, price in item_rows:
            cur.execute(
                "INSERT INTO order_items (order_id,product_id,quantity,unit_price) VALUES (%s,%s,%s,%s)",
                (order_id, pid, qty, price),
            )
        order_count += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nDone!")
    print(f"  customers   : {len(CUSTOMERS)}")
    print(f"  products    : {len(PRODUCTS)}")
    print(f"  orders      : {order_count}")
    print(f"  order_items : ~{order_count * 5} (avg 5 items/order)")

if __name__ == "__main__":
    main()
