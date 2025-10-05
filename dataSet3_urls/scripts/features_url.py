
import re
import pandas as pd
import tldextract
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "xyz","top","club","gq","tk","work","support","click","fit","pw","loan","kim","men","space"
}

def _count(pattern, s):
    return len(re.findall(pattern, s))

def _bool_int(x): 
    return 1 if x else 0

def _host(url):
    try: return urlparse(url).netloc
    except: return ""

def _path(url):
    try: return urlparse(url).path or "/"
    except: return "/"

def featurize_urls(urls):
    rows = []
    for u in urls:
        u_str = str(u)
        host = _host(u_str)
        path = _path(u_str)
        ext = tldextract.extract(host)
        features = {
            "len_url": len(u_str),
            "len_host": len(host),
            "len_path": len(path),
            "count_hyphen": u_str.count("-"),
            "count_at": u_str.count("@"),
            "count_question": u_str.count("?"),
            "count_equals": u_str.count("="),
            "count_digits": _count(r"[0-9]", u_str),
            "count_dots": u_str.count("."),
            "has_ip_host": _bool_int(bool(re.match(r"^(?:\d{1,3}\.){3}\d{1,3}$", host.split(":")[0]))),
            "host_num_labels": len(host.split(".")) if host else 0,
            "tld_len": len(ext.suffix) if ext.suffix else 0,
            "is_suspicious_tld": _bool_int(ext.suffix in SUSPICIOUS_TLDS),
            "contains_login": _bool_int("login" in u_str.lower()),
            "contains_verify": _bool_int("verify" in u_str.lower()),
            "contains_update": _bool_int("update" in u_str.lower()),
            "contains_secure": _bool_int("secure" in u_str.lower()),
            "contains_bank": _bool_int("bank" in u_str.lower()),
        }
        rows.append(features)
    return pd.DataFrame(rows)
