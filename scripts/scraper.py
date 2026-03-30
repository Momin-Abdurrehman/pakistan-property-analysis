"""
Zameen.com Property Listings Scraper
====================================
Scrapes property listings (Houses, Flats, Plots) from Zameen.com for major Pakistani cities.
Uses Playwright (headless Chromium) to handle Cloudflare JS challenges.

Usage:
    python scripts/scraper.py

Output:
    data/raw/zameen_raw_complete.csv

Note: For academic use only. Rate-limited to be respectful to the server.
"""

import csv
import os
import random
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

# ── Configuration ──────────────────────────────────────────────────────────────

# City name -> (zameen_id, max_pages_to_scrape)
CITIES = {
    "Lahore": (1, 120),
    "Karachi": (2, 120),
    "Islamabad": (3, 120),
    "Faisalabad": (16, 60),
    "Peshawar": (17, 60),
    "Rawalpindi": (41, 80),
}

# Property types to scrape: (URL slug, label for CSV)
PROPERTY_TYPES = [
    ("Homes", "House"),
    ("Flats_Apartments", "Flat"),
    ("Plots", "Plot"),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zameen_raw_complete.csv")

# Delay between page loads (seconds) — respectful rate limiting
MIN_DELAY = 2.0
MAX_DELAY = 5.0

CSV_COLUMNS = [
    "title",
    "price",
    "location",
    "city",
    "size",
    "bedrooms",
    "bathrooms",
    "property_type",
    "date_added",
    "url",
    "scraped_at",
]


# ── Extraction helpers ─────────────────────────────────────────────────────────

def extract_text(listing, aria_label):
    """Safely extract inner text from an element with given aria-label."""
    el = listing.locator(f'[aria-label="{aria_label}"]')
    if el.count() > 0:
        return el.first.inner_text().strip()
    return ""


def extract_listings(page, city_name, ptype_label):
    """Extract all listing data from the current page."""
    listings = page.locator('[aria-label="Listing"]').all()
    results = []

    for listing in listings:
        # Get the listing URL
        link_el = listing.locator('[aria-label="Listing link"]')
        href = ""
        if link_el.count() > 0:
            href = link_el.first.get_attribute("href") or ""
            if href and not href.startswith("http"):
                href = "https://www.zameen.com" + href

        row = {
            "title": extract_text(listing, "Title"),
            "price": extract_text(listing, "Price"),
            "location": extract_text(listing, "Location"),
            "city": city_name,
            "size": extract_text(listing, "Area"),
            "bedrooms": extract_text(listing, "Beds"),
            "bathrooms": extract_text(listing, "Baths"),
            "property_type": ptype_label,
            "date_added": extract_text(listing, "Listing creation date"),
            "url": href,
            "scraped_at": datetime.now().isoformat(),
        }
        results.append(row)

    return results


# ── Main scraper ───────────────────────────────────────────────────────────────

def scrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    total_scraped = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        for url_slug, ptype_label in PROPERTY_TYPES:
            for city_name, (city_id, max_pages) in CITIES.items():
                print(f"\n{'='*60}")
                print(f"Scraping {ptype_label} ({url_slug}) in {city_name} (ID: {city_id}, up to {max_pages} pages)")
                print(f"{'='*60}")

                consecutive_empty = 0

                for page_num in range(1, max_pages + 1):
                    url = f"https://www.zameen.com/{url_slug}/{city_name}-{city_id}-{page_num}.html"

                    try:
                        page.goto(url, timeout=30000, wait_until="domcontentloaded")
                        # Wait for listings to render
                        page.wait_for_selector('[aria-label="Listing"]', timeout=10000)
                    except Exception:
                        # Page might not have loaded or no listings
                        consecutive_empty += 1
                        if consecutive_empty >= 3:
                            print(f"  [!] 3 consecutive empty pages at page {page_num}, moving to next city")
                            break
                        delay = random.uniform(MIN_DELAY, MAX_DELAY)
                        time.sleep(delay)
                        continue

                    listings = extract_listings(page, city_name, ptype_label)

                    if not listings:
                        consecutive_empty += 1
                        if consecutive_empty >= 3:
                            print(f"  [!] 3 consecutive empty pages at page {page_num}, moving to next city")
                            break
                    else:
                        consecutive_empty = 0
                        all_data.extend(listings)
                        total_scraped += len(listings)

                    if page_num % 10 == 0 or page_num == 1:
                        print(f"  Page {page_num}: {len(listings)} listings (total so far: {total_scraped})")

                    # Respectful delay
                    delay = random.uniform(MIN_DELAY, MAX_DELAY)
                    time.sleep(delay)

                print(f"  Done with {city_name}: {total_scraped} total listings so far")

        browser.close()

    # Write to CSV
    print(f"\n{'='*60}")
    print(f"Writing {len(all_data)} listings to {OUTPUT_FILE}")
    print(f"{'='*60}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"Done! Saved {len(all_data)} listings to {OUTPUT_FILE}")
    return all_data


if __name__ == "__main__":
    scrape()
