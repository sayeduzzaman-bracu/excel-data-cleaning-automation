import os
import sys
import re
from datetime import datetime
from difflib import SequenceMatcher
from zipfile import BadZipFile

import pandas as pd

import warnings

warnings.filterwarnings(
    "ignore",
    message="Parsing dates in .* format when dayfirst=True was specified"
)

# ----------------------------
# CONFIG
# ----------------------------
INPUT_DIR = "input"
OUTPUT_DIR = "output"

FUZZY_THRESHOLD = 0.86
DATE_PARSE_THRESHOLD = 0.60
NUMERIC_PARSE_THRESHOLD = 0.75
PHONE_PARSE_THRESHOLD = 0.60

BLANK_LIKE_VALUES = {
    "", " ", "na", "n/a", "null", "none", "nil", "-", "--", "---", "nan"
}

DATE_COLUMN_HINTS = {
    "date", "joined", "join_date", "created", "created_at", "updated",
    "updated_at", "dob", "birth_date", "order_date", "invoice_date",
    "sale_date", "purchase_date", "restock_date", "last_restock", "time"
}

PHONE_COLUMN_HINTS = {
    "phone", "mobile", "contact", "contact_no", "contact_number", "phone_no",
    "phone_number", "cell", "cellphone", "tel", "telephone", "whatsapp"
}

EMAIL_COLUMN_HINTS = {
    "email", "email_address", "mail", "e_mail"
}

NAME_LIKE_HINTS = {
    "name", "customer", "customer_name", "full_name", "fullname", "client",
    "client_name", "supplier", "city", "product", "status", "membership",
    "warehouse"
}

ID_LIKE_HINTS = {
    "id", "order_id", "invoice_no", "invoice_number", "invoice", "sku",
    "client_id", "customer_id", "product_id", "transaction_id"
}


# ----------------------------
# Canonical column targets + synonyms
# Extend as needed
# ----------------------------
CANONICAL_SYNONYMS = {
    "phone": [
        "phone", "mobile", "contact", "contact_no", "contact_number", "phone_no",
        "phone_number", "cell", "cellphone", "tel", "telephone", "whatsapp",
        "supplier_phone"
    ],
    "name": [
        "name", "customer", "customer_name", "full_name", "fullname", "client",
        "client_name", "supplier", "vendor", "person_name"
    ],
    "quantity": [
        "quantity", "qty", "qnty", "quant", "no_of_items", "item_qty", "count",
        "units", "pcs", "pieces"
    ],
    "email": [
        "email", "email_address", "mail", "e_mail"
    ],
    "date": [
        "date", "join_date", "joined", "created_at", "updated_at",
        "order_date", "invoice_date", "last_restock", "restock_date"
    ],
}


# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_col_name(col: str) -> str:
    """
    Standardize column names:
    - strip
    - lowercase
    - whitespace -> underscore
    - remove non-alphanumeric/underscore
    - collapse underscores
    """
    if col is None:
        return ""
    s = str(col).strip().lower()
    s = s.replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_synonym_lookup(canonical_synonyms: dict) -> dict:
    lookup = {}
    for canonical, synonyms in canonical_synonyms.items():
        for syn in synonyms:
            lookup[normalize_col_name(syn)] = canonical
    return lookup


SYN_LOOKUP = build_synonym_lookup(CANONICAL_SYNONYMS)
CANONICAL_KEYS = list(CANONICAL_SYNONYMS.keys())


def map_similar_columns(columns):
    """
    Returns:
      mapped_columns: list of final column names
      rename_notes: list of (old, new, reason)
    """
    mapped = []
    notes = []

    for original in columns:
        std = normalize_col_name(original)

        # Exact synonym match
        if std in SYN_LOOKUP:
            new = SYN_LOOKUP[std]
            mapped.append(new)
            notes.append((str(original), new, f"synonym({std})"))
            continue

        # Fuzzy match vs canonical keys + synonyms
        best_score = 0.0
        best_target = None

        for target in CANONICAL_KEYS:
            sc = similarity(std, target)
            if sc > best_score:
                best_score = sc
                best_target = target

        for syn_norm, target in SYN_LOOKUP.items():
            sc = similarity(std, syn_norm)
            if sc > best_score:
                best_score = sc
                best_target = target

        if best_target is not None and best_score >= FUZZY_THRESHOLD:
            mapped.append(best_target)
            notes.append((str(original), best_target, f"fuzzy({std}~{best_score:.2f})"))
        else:
            mapped.append(std)
            if str(original) != std:
                notes.append((str(original), std, "standardized"))

    return mapped, notes


def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def collapse_internal_spaces(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).replace("\n", " ").replace("\t", " ").strip()
    s = re.sub(r"\s+", " ", s)
    if s.lower() in BLANK_LIKE_VALUES:
        return pd.NA
    return s


def normalize_blank_like(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(collapse_internal_spaces)
    return df


def strip_excel_float_text(value: str) -> str:
    """
    Turns '123.0' into '123' for text-like identifiers/phones.
    """
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    if re.fullmatch(r"\d+\.0", s):
        return s[:-2]
    return s


def normalize_email(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().lower()
    if s in BLANK_LIKE_VALUES:
        return pd.NA
    return s


def normalize_phone(value):
    """
    Generic phone cleaning:
    - preserve leading +
    - remove spaces, hyphens, parentheses, dots
    - keep as text
    """
    if pd.isna(value):
        return pd.NA

    s = strip_excel_float_text(value).strip()
    if not s or s.lower() in BLANK_LIKE_VALUES:
        return pd.NA

    s = s.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    s = re.sub(r"[^0-9+]", "", s)

    # Allow only one leading +
    if s.count("+") > 1:
        s = s.replace("+", "")
    if "+" in s and not s.startswith("+"):
        s = s.replace("+", "")

    return s if s else pd.NA


def looks_like_phone_series(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return False

    cleaned = s.map(normalize_phone)
    valid = cleaned.dropna()

    if valid.empty:
        return False

    ratio = len(valid) / len(s)
    return ratio >= PHONE_PARSE_THRESHOLD


def title_case_preserving_acronyms(value):
    if pd.isna(value):
        return pd.NA

    s = str(value).strip()
    if not s:
        return pd.NA

    words = s.split(" ")
    out = []
    for w in words:
        if w.isupper() and len(w) <= 4 and any(ch.isalpha() for ch in w):
            out.append(w)
        else:
            out.append(w.lower().capitalize())
    return " ".join(out)


def maybe_normalize_text_column(col_name: str, series: pd.Series) -> pd.Series:
    """
    Title-case only for human-readable text-like columns.
    Avoid IDs, emails, phones, very numeric columns.
    """
    col_norm = normalize_col_name(col_name)

    if any(hint in col_norm for hint in ID_LIKE_HINTS):
        return series

    if any(hint in col_norm for hint in PHONE_COLUMN_HINTS):
        return series

    if any(hint in col_norm for hint in EMAIL_COLUMN_HINTS):
        return series

    if col_norm in NAME_LIKE_HINTS or any(hint in col_norm for hint in NAME_LIKE_HINTS):
        return series.map(title_case_preserving_acronyms)

    return series


def try_parse_single_date(value, dayfirst=False):
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip()
    if not s:
        return pd.NaT

    known_formats = [
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%y/%m/%d",
    ]

    for fmt in known_formats:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            pass

    try:
        return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    except Exception:
        return pd.NaT


def parse_mixed_date_series(series: pd.Series):
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return None, 0.0

    parsed_a = series.map(lambda x: try_parse_single_date(x, dayfirst=False))
    parsed_b = series.map(lambda x: try_parse_single_date(x, dayfirst=True))

    ratio_a = parsed_a.notna().mean()
    ratio_b = parsed_b.notna().mean()

    if ratio_b > ratio_a:
        return parsed_b, ratio_b
    return parsed_a, ratio_a

def is_date_column(col_name: str, series: pd.Series) -> bool:
    col_norm = normalize_col_name(col_name)
    s = series.dropna()

    if s.empty:
        return False

    if col_norm in DATE_COLUMN_HINTS or any(hint in col_norm for hint in DATE_COLUMN_HINTS):
        _, ratio = parse_mixed_date_series(series.astype(str))
        return ratio >= 0.30  # lower threshold if header strongly hints date

    # heuristic detection
    _, ratio = parse_mixed_date_series(series.astype(str))
    return ratio >= DATE_PARSE_THRESHOLD


def normalize_date_column(series: pd.Series) -> pd.Series:
    parsed, _ = parse_mixed_date_series(series.astype(str))
    if parsed is None:
        return series
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), pd.NA)


def looks_like_numeric_text(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.strip()
    if s.empty:
        return False

    def clean_num(x):
        x = x.replace(",", "")
        x = x.replace("$", "").replace("€", "").replace("£", "").replace("৳", "")
        x = x.replace("%", "")
        return x

    cleaned = s.map(clean_num)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    ratio = parsed.notna().mean()
    return ratio >= NUMERIC_PARSE_THRESHOLD


def is_id_like_column(col_name: str) -> bool:
    col_norm = normalize_col_name(col_name)
    return col_norm in ID_LIKE_HINTS or any(hint in col_norm for hint in ID_LIKE_HINTS)


def normalize_numeric_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    def clean_num(x):
        if x.lower() in BLANK_LIKE_VALUES:
            return pd.NA
        x = x.replace(",", "")
        x = x.replace("$", "").replace("€", "").replace("£", "").replace("৳", "")
        x = x.replace("%", "")
        return x

    cleaned = s.map(clean_num)
    num = pd.to_numeric(cleaned, errors="coerce")

    # If all non-null numeric values are whole numbers, store as Int64
    non_null = num.dropna()
    if not non_null.empty and (non_null % 1 == 0).all():
        return num.astype("Int64")

    return num.astype("Float64")


def read_table_file(path_in: str) -> pd.DataFrame:
    """
    Smart reader that checks actual file content, not just extension.

    Handles:
    - real CSV
    - real XLSX/XLSM
    - real XLS
    - CSV renamed as .xlsx
    - XLSX renamed as .csv
    """

    ext = os.path.splitext(path_in)[1].lower()

    def read_csv_fallback(p):
        try:
            return pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(p, encoding="cp1252")
            except UnicodeDecodeError:
                return pd.read_csv(p, encoding="latin1")

    def is_zip_excel(p):
        # XLSX/XLSM files are ZIP-based
        try:
            with open(p, "rb") as f:
                sig = f.read(4)
            return sig == b"PK\x03\x04"
        except Exception:
            return False

    def is_legacy_xls(p):
        # Old XLS files use OLE compound file signature
        try:
            with open(p, "rb") as f:
                sig = f.read(8)
            return sig == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        except Exception:
            return False

    # Detect by file signature first
    if is_zip_excel(path_in):
        try:
            return pd.read_excel(path_in, engine="openpyxl")
        except Exception as e:
            print(f"⚠️ Detected ZIP-based Excel file but failed to read with openpyxl: {e}")
            raise

    if is_legacy_xls(path_in):
        try:
            return pd.read_excel(path_in, engine="xlrd")
        except Exception as e:
            print(f"⚠️ Detected legacy XLS file but failed to read with xlrd: {e}")
            raise

    # If not detected as Excel by signature, try CSV
    try:
        return read_csv_fallback(path_in)
    except Exception as csv_error:
        # Last-resort fallback by extension
        try:
            if ext in [".xlsx", ".xlsm"]:
                return pd.read_excel(path_in, engine="openpyxl")
            if ext == ".xls":
                return pd.read_excel(path_in, engine="xlrd")
        except Exception:
            pass

        raise ValueError(
            f"Could not read file '{path_in}'. "
            f"It does not appear to be a valid CSV/XLSX/XLS file. "
            f"CSV error: {csv_error}"
        )

def pick_first_data_file(input_dir: str):
    files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".xlsx", ".xlsm", ".xls", ".csv"))
    ]
    files.sort()
    return files[0] if files else None


def auto_fit_excel_columns(path_out: str):
    """
    Auto-adjust exported Excel column widths.
    """
    try:
        from openpyxl import load_workbook

        wb = load_workbook(path_out)
        ws = wb.active

        for col_cells in ws.columns:
            max_len = 0
            col_letter = col_cells[0].column_letter
            for cell in col_cells:
                value = "" if cell.value is None else str(cell.value)
                if len(value) > max_len:
                    max_len = len(value)

            ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)

        wb.save(path_out)
    except Exception as e:
        print(f"⚠️ Could not auto-fit Excel columns: {e}")


def clean_values_by_column(df: pd.DataFrame):
    """
    Universal-ish value cleaning across all columns.
    Returns cleaned df + notes.
    """
    notes = []

    for col in df.columns:
        col_norm = normalize_col_name(col)

        # Skip fully empty columns
        if df[col].dropna().empty:
            continue

        # Work on object/string-like columns first
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # Email columns
            if col_norm in EMAIL_COLUMN_HINTS or any(h in col_norm for h in EMAIL_COLUMN_HINTS):
                df[col] = df[col].map(normalize_email)
                notes.append(f"{col}: normalized email values")
                continue

            # Phone columns
            if col_norm in PHONE_COLUMN_HINTS or any(h in col_norm for h in PHONE_COLUMN_HINTS):
                df[col] = df[col].map(normalize_phone)
                notes.append(f"{col}: normalized phone values")
                continue

            # Date columns
            if is_date_column(col, df[col]):
                old_non_null = df[col].notna().sum()
                df[col] = normalize_date_column(df[col])
                new_non_null = df[col].notna().sum()
                if new_non_null > 0:
                    notes.append(f"{col}: normalized dates to YYYY-MM-DD ({new_non_null}/{old_non_null} parsed)")
                continue

            # ID-like columns stay text
            if is_id_like_column(col):
                df[col] = df[col].map(lambda x: strip_excel_float_text(x) if not pd.isna(x) else pd.NA)
                continue

            # Numeric-like text columns
            if looks_like_numeric_text(df[col]) and not looks_like_phone_series(df[col]):
                df[col] = normalize_numeric_column(df[col])
                notes.append(f"{col}: normalized numeric text to numeric dtype")
                continue

            # Text normalization
            df[col] = maybe_normalize_text_column(col, df[col])

        else:
            # Native numeric/datetime columns
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d").where(df[col].notna(), pd.NA)
                notes.append(f"{col}: normalized datetime dtype to YYYY-MM-DD")

    return df, notes


def clean_file(path_in: str) -> str:
    df = read_table_file(path_in)
    rows_before = len(df)

    # Standardize + map columns
    mapped_cols, rename_notes = map_similar_columns(df.columns)
    df.columns = make_unique_columns(mapped_cols)

    # Drop fully empty rows first
    df = df.dropna(how="all")

    # Normalize blank-like strings + trim
    df = normalize_blank_like(df)

    # Remove fully empty columns too
    df = df.dropna(axis=1, how="all")

    # Universal value cleaning
    df, value_notes = clean_values_by_column(df)

    # Remove rows where canonical name+phone are both missing
    if "name" in df.columns and "phone" in df.columns:
        df = df.dropna(subset=["name", "phone"], how="all")

    # Exact duplicate removal
    rows_before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = rows_before_dedup - len(df)

    rows_after = len(df)
    missing_per_col = df.isna().sum().sort_values(ascending=False)

    # Export
    base = os.path.splitext(os.path.basename(path_in))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{base}_cleaned_{ts}.xlsx"
    path_out = os.path.join(OUTPUT_DIR, out_name)

    df.to_excel(path_out, index=False)
    auto_fit_excel_columns(path_out)

    # Summary
    print("\n🧼 Cleaner Summary")
    print("-" * 60)
    print(f"Input:              {path_in}")
    print(f"Output:             {path_out}")
    print(f"Rows before:        {rows_before}")
    print(f"Rows after:         {rows_after}")
    print(f"Duplicates removed: {duplicates_removed}")
    print()

    changes = []
    for old, new, reason in rename_notes:
        old_std = normalize_col_name(old)
        if old_std != new or "fuzzy" in reason or "synonym" in reason:
            changes.append((old, new, reason))

    if changes:
        print("Column mapping:")
        for old, new, reason in changes:
            print(f"  - {old!r} -> {new!r}   [{reason}]")
        print()

    if value_notes:
        print("Value normalization:")
        for note in value_notes:
            print(f"  - {note}")
        print()

    print("Missing values per column (only > 0):")
    has_missing = missing_per_col[missing_per_col > 0]
    if len(has_missing) == 0:
        print("  (none 🎉)")
    else:
        for col, cnt in has_missing.items():
            print(f"  - {col}: {int(cnt)}")

    print("-" * 60)
    return path_out


def main():
    ensure_dirs()

    # Usage:
    #   python cleaner.py
    #   python cleaner.py input/file
    #   python cleaner.py file.xlsx
    if len(sys.argv) >= 2:
        candidate = sys.argv[1]
        if os.path.isfile(candidate):
            input_path = candidate
        else:
            input_path = os.path.join(INPUT_DIR, candidate)
    else:
        first = pick_first_data_file(INPUT_DIR)
        if not first:
            print(f"Put a file (.xlsx/.xlsm/.xls/.csv) inside ./{INPUT_DIR}/ and run again.")
            print("Example: python cleaner.py")
            sys.exit(1)
        input_path = os.path.join(INPUT_DIR, first)

    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    clean_file(input_path)


if __name__ == "__main__":
    main()