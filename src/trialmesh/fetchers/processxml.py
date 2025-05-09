#!/usr/bin/env python3
# src/trialmesh/data/processxml.py

import argparse
import glob
import json
import logging
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from multiprocessing import Pool, cpu_count
import csv
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process SIGIR2016 clinical trial XML files")
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Base directory containing raw dataset files (default: ./data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for saving processed files; defaults to {data-dir}/sigir2016/processed')
    parser.add_argument('--log-level', default='WARNING', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set logging verbosity level (default: WARNING)')
    args = parser.parse_args()

    # Set default output dir if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'sigir2016', 'processed')

    # Ensure output directory exists for log file
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def setup_logging(args):
    """Set up logging configuration based on user arguments.

    This function configures both file and console logging with appropriate
    formatting and log levels.

    Args:
        args: Command-line arguments namespace
    """
    # Ensure log directory exists first
    log_dir = Path(args.data_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"processxml_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    console_level = getattr(logging, args.log_level)

    # Set up file handler (always DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Set up console handler (level based on user input)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # This ensures all messages are processed
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Console logging level set to: {logging.getLevelName(console_level)}")
    logging.debug(f"File logging level set to: DEBUG, log file: {log_file}")


def time_function(func_name, start_time):
    """Log the execution time of a function if in debug mode.

    This utility function calculates and logs the execution time of
    a function, but only if the logging level is set to DEBUG or lower.

    Args:
        func_name: Name of the function being timed
        start_time: Start time of the function execution
    """
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        end_time = time.time()
        logging.debug(f"{func_name} took {end_time - start_time:.2f} seconds to execute.")


def clean_text(text):
    """Clean and normalize text content.

    This function performs several text normalization steps:
    1. Removing excess whitespace
    2. Replacing Unicode characters with ASCII equivalents
    3. Normalizing special symbols

    Args:
        text: Text to clean

    Returns:
        Cleaned text string
    """
    if text is None:
        return ""
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace Unicode characters with ASCII equivalents
    text = text.replace('\u2264', '<=').replace('\u2265', '>=')
    return text


def process_xml_file(file_path):
    """Extract relevant data from a clinical trial XML file.

    This function parses a clinical trial XML document and extracts
    structured information about the trial, including:
    - Trial identifier (NCT ID)
    - Title
    - Summary
    - Detailed description
    - Inclusion and exclusion criteria
    - Enrollment information
    - Interventions
    - Conditions
    - Phase information

    Args:
        file_path: Path to the XML file

    Returns:
        Dictionary with extracted trial data, or None if processing failed
    """
    # Skip macOS metadata files
    if os.path.basename(file_path).startswith('._'):
        return None

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        nct_id = root.find('id_info/nct_id').text
        title = clean_text(root.find('brief_title').text)

        brief_summary = root.find('brief_summary/textblock')
        brief_summary_text = clean_text(brief_summary.text) if brief_summary is not None else ""

        detailed_description = root.find('detailed_description/textblock')
        detailed_description_text = clean_text(detailed_description.text) if detailed_description is not None else ""

        criteria = root.find('.//criteria/textblock')
        criteria_text = clean_text(criteria.text) if criteria is not None else ""

        # Split criteria text into inclusion and exclusion
        inclusion_text = ""
        exclusion_text = ""
        if "Inclusion Criteria:" in criteria_text and "Exclusion Criteria:" in criteria_text:
            parts = criteria_text.split("Exclusion Criteria:")
            inclusion_text = parts[0].split("Inclusion Criteria:")[-1].strip()
            exclusion_text = parts[1].strip()
        elif "Inclusion Criteria:" in criteria_text:
            inclusion_text = criteria_text.split("Inclusion Criteria:")[-1].strip()
        elif "Exclusion Criteria:" in criteria_text:
            exclusion_text = criteria_text.split("Exclusion Criteria:")[-1].strip()

        # Clean up inclusion and exclusion criteria
        inclusion_text = clean_text(inclusion_text)
        exclusion_text = clean_text(exclusion_text)

        enrollment = root.find('enrollment')
        enrollment_value = enrollment.text if enrollment is not None else "0"

        intervention_elements = root.findall('.//intervention/intervention_name')
        drugs_list = [clean_text(elem.text) for elem in intervention_elements if elem.text]

        condition_elements = root.findall('condition')
        diseases_list = [clean_text(elem.text) for elem in condition_elements if elem.text]

        phase_element = root.find('phase')
        phase = clean_text(phase_element.text) if phase_element is not None else ""

        data = {
            "_id": nct_id,
            "title": title,
            "metadata": {
                "phase": phase,
                "drugs": str(drugs_list),
                "drugs_list": drugs_list,
                "diseases_list": diseases_list,
                "enrollment": enrollment_value,
                "inclusion_criteria": inclusion_text,
                "exclusion_criteria": exclusion_text,
                "brief_summary": brief_summary_text,
                "detailed_description": detailed_description_text
            }
        }

        return data

    except ET.ParseError:
        logging.error(f"Invalid XML file: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return None


def copy_and_transform_files(data_dir, output_dir):
    """Transform and copy auxiliary files from the dataset.

    This function handles the conversion of supporting files from the
    SIGIR2016 dataset into formats used by TrialMesh. Specifically:
    1. Transforms adhoc-queries.json to queries.jsonl format
    2. Transforms qrels-clinical_trials.txt to test.tsv format

    Args:
        data_dir: Base data directory
        output_dir: Output directory for transformed files
    """
    start_time = time.time()

    source_dir = os.path.join(data_dir, "sigir2016", "data")
    files_to_copy = [
        ("adhoc-queries.json", "queries.jsonl"),
        ("qrels-clinical_trials.txt", "test.tsv")
    ]

    for source_file, dest_file in files_to_copy:
        source_path = os.path.join(source_dir, source_file)
        destination_path = os.path.join(output_dir, dest_file)

        try:
            if source_file == "adhoc-queries.json":
                # Read the original JSON file
                with open(source_path, 'r', encoding='utf-8') as f:
                    queries = json.load(f)

                # Transform the queries and write to JSONL format
                with open(destination_path, 'w', encoding='utf-8') as f:
                    for query in queries:
                        transformed_query = {
                            "_id": f"sigir-{query['qId'][4:].replace('-', '')}",
                            "text": query['description']
                        }
                        json.dump(transformed_query, f, ensure_ascii=False)
                        f.write('\n')

                logging.info(f"Successfully transformed {source_file} and saved as {dest_file} in {output_dir}")

            elif source_file == "qrels-clinical_trials.txt":
                # Read the original file and transform
                with open(source_path, 'r', encoding='utf-8') as infile, \
                        open(destination_path, 'w', encoding='utf-8', newline='') as outfile:

                    tsv_writer = csv.writer(outfile, delimiter='\t')

                    # Write the header
                    tsv_writer.writerow(['query-id', 'corpus-id', 'score'])

                    # Process and write the data
                    for line in infile:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            query_id = f"sigir-{parts[0]}"
                            corpus_id = parts[2]
                            score = parts[3]
                            tsv_writer.writerow([query_id, corpus_id, score])

                logging.info(f"Successfully transformed {source_file} and saved as {dest_file} in {output_dir}")

            else:
                # For any other files, just copy without modification
                shutil.copy2(source_path, destination_path)
                logging.info(f"Successfully copied {source_file} to {output_dir} as {dest_file}")

        except FileNotFoundError:
            logging.error(f"Source file not found: {source_path}")
        except PermissionError:
            logging.error(f"Permission denied when copying {source_file}")
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON in {source_file}")
        except Exception as e:
            logging.error(f"Error processing {source_file}: {str(e)}")

    time_function("copy_and_transform_files", start_time)


def main():
    """Main entry point for the script.

    This function orchestrates the entire XML processing pipeline:
    1. Setting up directories and logging
    2. Finding XML files to process
    3. Processing the files in parallel
    4. Writing the results to JSONL
    5. Transforming auxiliary files
    """
    main_start_time = time.time()

    args = parse_args()
    setup_logging(args)

    # Convert string paths to Path objects for easier manipulation
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Clean and recreate output directory
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Removed and recreated directory: {output_dir}")

    # Find the XML files in the extracted dataset directory
    trials_dir = data_dir / "sigir2016" / "data" / "clinicaltrials.gov-16_dec_2015"

    # If the first path doesn't exist, try the nested structure that might be generated by extraction
    if not trials_dir.exists():
        nested_trials_dir = trials_dir / "clinicaltrials.gov-16_dec_2015"
        if nested_trials_dir.exists():
            trials_dir = nested_trials_dir
        else:
            logging.error(f"Trial directory not found at {trials_dir} or {nested_trials_dir}")
            logging.error("Please ensure the SIGIR2016 dataset has been downloaded and extracted correctly.")
            return

    # Filter out the macOS metadata files (._*) when collecting XML files
    xml_files = [f for f in trials_dir.glob("*.xml") if not f.name.startswith('._')]

    if not xml_files:
        logging.error(
            f"No XML files found in {trials_dir}. Please ensure the SIGIR2016 dataset has been downloaded correctly.")
        return

    logging.info(f"Found {len(xml_files)} XML files to process")

    # Process XML files in parallel
    processing_start_time = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_xml_file, xml_files)
    time_function("XML processing", processing_start_time)

    valid_results = [result for result in results if result is not None]

    # Write the processed trials to a JSONL file
    write_start_time = time.time()
    output_file = output_dir / "corpus.jsonl"
    with open(output_file, 'w') as f:
        for result in valid_results:
            json.dump(result, f)
            f.write('\n')
    time_function("Writing results", write_start_time)

    logging.info(f"Processed {len(valid_results)} files successfully. Output saved to {output_file}")

    # Copy and transform auxiliary files
    copy_and_transform_files(data_dir, output_dir)

    time_function("Total processing", main_start_time)


def cli_main():
    """Entry point for the console script"""
    main()


if __name__ == "__main__":
    main()