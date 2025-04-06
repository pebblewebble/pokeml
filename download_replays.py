from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from bs4 import BeautifulSoup
import os
import time
import re
import argparse

# --- Helper Function to Extract ID from href ---
def extract_id_from_href(href_str, format_str):
    """Extracts the numerical ID from href like 'gen9ou-12345' or '/gen9ou-12345'"""
    # Regex: Match optional leading slash, then the format string, hyphen, and digits.
    # Use re.escape on format_str to handle potential special characters in format names.
    pattern = rf'/?{re.escape(format_str)}-(\d+)' # Allow an OPTIONAL leading slash '?'
    match = re.search(pattern, href_str)
    if match:
        return match.group(1) # Return the captured digits (the ID)
    # Optional: Add a print here if debugging is still needed
    # print(f"  DEBUG: Regex failed for href='{href_str}', pattern='{pattern}'")
    return None

# --- Function to Scrape IDs from Listing Pages (Further Debugging) ---
def scrape_replay_ids(target_format, num_ids_to_find):
    base_list_url = "https://replay.pokemonshowdown.com/"
    found_ids = set()
    current_page = 1

    print(f"Starting scrape for {num_ids_to_find} IDs for format '{target_format}' using Selenium...")

    # --- Configure Selenium WebDriver (same setup logic as before) ---
    try:
        from selenium.webdriver.chrome.service import Service as ChromeService
        from webdriver_manager.chrome import ChromeDriverManager
        print("  Attempting to use webdriver-manager for ChromeDriver...")
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
        print("  ChromeDriver started via webdriver-manager.")
    except ImportError:
        print("  webdriver-manager not found. Trying default driver location...")
        try:
             driver = webdriver.Chrome()
             print("  ChromeDriver started from default location.")
        except Exception as e_chrome:
            print(f"  Failed to start Chrome ({e_chrome}). Trying Firefox...")
            try:
                driver = webdriver.Firefox()
                print("  Firefox/GeckoDriver started from default location.")
            except Exception as e_firefox:
                print(f"  ERROR: Could not automatically start Chrome or Firefox WebDriver: {e_firefox}")
                return set()
    # ------------------------------------

    while len(found_ids) < num_ids_to_find:
        page_url = f"{base_list_url}?format={target_format}"
        if current_page > 1:
            page_url += f"&page={current_page}"

        print(f"\nScraping page {current_page}: {page_url}")

        try:
            driver.get(page_url)
            wait_time = 10
            print(f"  Waiting up to {wait_time}s for element <ul class='linklist'> li a to load...") # Wait for a link inside the list

            try:
                 # --- MODIFIED WAIT: Wait for the first LINK *inside* the list ---
                 # This ensures not just the container, but also its content is likely present.
                WebDriverWait(driver, wait_time).until(
                    # EC.presence_of_element_located((By.CSS_SELECTOR, "ul.linklist")) # Old wait
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ul.linklist li a")) # New: wait for first link *within* a list item in the ul
                )
                print("  DEBUG: Found at least one link inside <ul class='linklist'> via Selenium wait.")
            except TimeoutException:
                print(f"  Error: Timed out waiting for links within <ul class='linklist'> on page {current_page}.")
                if current_page > 1: print("  Assuming end of replays.")
                else: print("  Failed to load list content on page 1.")
                break
            except NoSuchElementException:
                 print(f"  Error: Could not find links within <ul class='linklist'> even after waiting.")
                 break

            # --- Get page source and parse ---
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'lxml')
            replay_list = soup.find('ul', class_='linklist')

            if not replay_list:
                 print("  ERROR: Found links with Selenium wait, but couldn't find parent <ul class='linklist'> in parsed source!")
                 break

            # --- DEBUG: Print the content of the found list ---
            print("--------------------------------------------------")
            print(f"  DEBUG: Content of found <ul class='linklist'> (first 1000 chars):")
            print(str(replay_list)[:1000])
            print("--------------------------------------------------")
            # --------------------------------------------------

            links = replay_list.find_all('a')
            ids_on_page = 0
            print(f"  DEBUG: Found {len(links)} <a> tags within the list.") # DEBUG how many links found

            for i, link in enumerate(links):
                href = link.get('href')
                # --- DEBUG: Print each link being processed ---
                print(f"    Processing link {i+1}/{len(links)} - Href: {href}")
                # ---------------------------------------------
                if href:
                    replay_id = extract_id_from_href(href, target_format)
                    if replay_id:
                        print(f"      Extracted ID: {replay_id}") # DEBUG successful extraction
                        if replay_id not in found_ids:
                            found_ids.add(replay_id)
                            ids_on_page += 1
                        if len(found_ids) >= num_ids_to_find:
                            break
                    #else: # Optional DEBUG if extraction fails
                    #    print(f"      Failed to extract ID from href: {href}")


            print(f"Found {ids_on_page} new unique IDs on page {current_page}. Total found: {len(found_ids)}")

            # --- Stopping conditions (same as before) ---
            if ids_on_page == 0 and current_page > 1:
                 print(f"No new IDs found on page {current_page}. Assuming end of replays.")
                 break
            if len(found_ids) >= num_ids_to_find:
                print(f"Target number of IDs ({num_ids_to_find}) reached.")
                break

            current_page += 1

        except Exception as e:
            print(f"An unexpected error occurred during Selenium scraping on page {current_page}: {e}")
            import traceback
            traceback.print_exc()
            break

    # --- Clean up WebDriver (same as before) ---
    try:
        driver.quit()
        print("WebDriver closed.")
    except Exception as e_quit:
        print(f"Note: Error closing WebDriver: {e_quit}")

    print(f"\nFinished scraping. Found {len(found_ids)} unique IDs.")
    return found_ids

# --- download_replays_by_id function (no changes needed) ---
def download_replays_by_id(replay_ids, format_str, output_dir):
    """Downloads Pokémon Showdown replay logs for the given IDs."""
    base_log_url = "https://replay.pokemonshowdown.com/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nStarting download of {len(replay_ids)} log files...")
    print(f"Output directory: {output_dir}")

    downloaded_count = 0
    attempted_count = 0
    total_ids = len(replay_ids)

    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0'}

    for replay_id_str in replay_ids:
        attempted_count += 1
        log_filename = f"{format_str}-{replay_id_str}.log"
        log_url = f"{base_log_url}{log_filename}"
        output_path = os.path.join(output_dir, log_filename)

        print(f"\nAttempting download ({attempted_count}/{total_ids}): ID {replay_id_str}")
        print(f"Log URL: {log_url}")

        if os.path.exists(output_path):
            print(f"Skipping: File already exists at {output_path}")
            continue

        try:
            response = session.get(log_url, headers=headers, timeout=15)
            if response.status_code == 200 and '|teamsize|' in response.text[:1000]:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Success: Saved to {output_path}")
                downloaded_count += 1
            elif response.status_code == 404:
                 print(f"Failed: Log file not found (404).")
            else:
                 print(f"Failed: Received status code {response.status_code} or content doesn't look like a log file.")

        except requests.exceptions.RequestException as e:
            print(f"Error: Network or request error for {log_url}: {e}")

        time.sleep(0.5)

    print(f"\nLog download process finished.")
    print(f"Attempted to download: {attempted_count}")
    print(f"Successfully downloaded (this run): {downloaded_count}")


# --- __main__ block (no changes needed) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Pokemon Showdown replay IDs and download logs.")

    parser.add_argument("-n", "--num_replays", type=int, default=50,
                        help="Target number of unique replay IDs to find and download.")
    parser.add_argument("-f", "--format", type=str, default="gen9ou",
                        help="The Pokemon Showdown format string (e.g., gen9ou).")
    parser.add_argument("-o", "--output_dir", type=str, default="replay_logs",
                        help="Directory to save the downloaded .log files.")

    args = parser.parse_args()

    # Step 1: Scrape the IDs
    found_ids = scrape_replay_ids(
        target_format=args.format,
        num_ids_to_find=args.num_replays
    )

    # Step 2: Download the logs for the found IDs
    if found_ids:
        download_replays_by_id(
            replay_ids=found_ids,
            format_str=args.format,
            output_dir=args.output_dir
        )
    else:
        print("\nNo replay IDs were found or an error occurred during scraping, skipping download.")