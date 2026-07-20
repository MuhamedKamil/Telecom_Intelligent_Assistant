import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from datetime import datetime
import time
import hashlib
import random
from urllib.parse import urlparse

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0"
]

def process_300_links(
    input_folder,
    output_folder=None,
    screenshots_folder=None,
    max_retries=2,          # Increased retries to handle the recovery step
    full_page=True,
    viewport_width=1920,
    viewport_height=1080,
    timeout=45000           
):
    input_path = Path(input_folder)
    output_path = Path(output_folder) if output_folder else input_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    screenshots_path = Path(screenshots_folder) if screenshots_folder else input_path / "screenshots"
    screenshots_path.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_path.glob("*.json"))
    total_files = len(json_files)
    
    print(f"🚀 Starting automated processing for {total_files} items.")
    
    launch_args = [
        '--disable-dev-shm-usage',
        '--no-sandbox',
        '--disable-blink-features=AutomationControlled', 
        '--disable-web-security'
    ]
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=launch_args)
        
        for idx, json_file in enumerate(json_files, 1):
            print(f"\n🔗 [{idx}/{total_files}] Processing file: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                url = data.get('url')
                if not url or not urlparse(url).scheme:
                    print("  ⚠️ Skip: URL missing or broken.")
                    continue
                
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                screenshot_path = screenshots_path / f"{json_file.stem}_{url_hash}.png"
                
                # Dynamic pacing delay between successful visits
                pacing_delay = random.uniform(5.0, 10.0)
                print(f"  ⏳ Mimicking human delay... waiting {pacing_delay:.1f}s")
                time.sleep(pacing_delay)
                
                screenshot_success = False
                
                for attempt in range(max_retries + 1):
                    context = browser.new_context(
                        viewport={"width": viewport_width, "height": viewport_height},
                        user_agent=random.choice(USER_AGENTS),
                        locale="en-US"
                    )
                    
                    page = context.new_page()
                    page.set_default_timeout(timeout)
                    
                    try:
                        if attempt > 0:
                            print(f"  🔄 Retry attempt {attempt}...")
                        
                        page.goto(url, wait_until="commit", timeout=timeout)
                        time.sleep(2)
                        
                        page.screenshot(path=str(screenshot_path), full_page=full_page, timeout=20000)
                        print("  ✅ Captured successfully.")
                        
                        data['screenshot'] = str(screenshot_path)
                        screenshot_success = True
                        break
                        
                    except PlaywrightTimeoutError:
                        print(f"  🛑 Website blocked us (Timeout hit on attempt {attempt}).")
                        
                        # Only sleep if we have remaining retries left for this file
                        if attempt < max_retries:
                            print("  ⏳ TRIGGERING COOL-DOWN: Sleeping for 5.5 minutes to let the website unblock our IP...")
                            # 330 seconds = 5.5 minutes (gives a 30-second safety window over the 5-minute block)
                            time.sleep(330) 
                            print("  🔄 Resuming processing now...")
                        else:
                            print("  ❌ Max retries reached for this link. Moving to next file.")
                            
                    except Exception as e:
                        print(f"  ❌ Error: {str(e)}")
                    finally:
                        page.close()
                        context.close()
                
                if not screenshot_success:
                    data['screenshot'] = "ERROR: Capture failed due to website temporary lock"
                
                with open(output_path / json_file.name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    
            except Exception as file_error:
                print(f"💥 Failed processing file structural contents: {str(file_error)}")
                
        browser.close()
    print("\n🏁 Task completely completed.")

if __name__ == "__main__":
    INPUT_FOLDER = "./WebData" 
    OUTPUT_FOLDER = "./updated_jsons"
    SCREENSHOTS_FOLDER = "./output_screenshots"

    process_300_links(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        screenshots_folder=SCREENSHOTS_FOLDER
    )
