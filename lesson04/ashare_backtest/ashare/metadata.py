import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
from string import Template

import pandas as pd


def license_uri(path):
    values = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"(name|password|host|port|url)=(.*)", raw_line.strip())
        if match:
            key, value = match.groups()
            values[key] = Template(value.strip().strip("\"'")).safe_substitute(values)
    if not {"name", "password", "host", "port", "url"}.issubset(values) or "$" in values["url"]:
        raise ValueError("无法安全解析本机 License；不会执行脚本")
    return values["url"]


def fetch(source, destination, start, end, license_script=None):
    import json
    import warnings
    import rqdatac
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    symbols = pd.read_feather(Path(source) / "close.feather").columns.drop("date").tolist()
    mapping = {symbol.replace(".SH", ".XSHG").replace(".SZ", ".XSHE"): symbol for symbol in symbols}
    original = os.environ.get("RQDATAC2_CONF")
    if license_script:
        os.environ["RQDATAC2_CONF"] = license_uri(license_script)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rqdatac.init()
        for field, function in (("is_st", rqdatac.is_st_stock), ("suspended", rqdatac.is_suspended)):
            batches = []
            for offset in range(0, len(mapping), 200):
                batch = list(mapping)[offset:offset + 200]
                frame = function(batch, start_date=start, end_date=end)
                if frame is None or frame.empty:
                    raise ValueError(f"{field} 数据返回为空")
                batches.append(frame.rename(columns=mapping))
                print(f"{field}: {min(offset + 200, len(mapping))}/{len(mapping)}", flush=True)
            pd.concat(batches, axis=1).reindex(columns=symbols).rename_axis("date").reset_index().to_feather(destination / f"{field}.feather")
        records = []
        for offset in range(0, len(mapping), 200):
            instruments = rqdatac.instruments(list(mapping)[offset:offset + 200])
            for instrument in instruments:
                if instrument is not None:
                    record = {"symbol": mapping[instrument.order_book_id]}
                    for field in ("listed_date", "de_listed_date"):
                        value = getattr(instrument, field, None)
                        record[{"listed_date": "list_date", "de_listed_date": "delist_date"}[field]] = value if value and str(value) != "0000-00-00" else None
                    records.append(record)
        pd.DataFrame(records).to_csv(destination / "instruments.csv", index=False)
        manifest = {"provider": "RQData", "api": ["is_st_stock", "is_suspended", "instruments"],
                    "start": start, "end": end, "symbols_requested": len(symbols),
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "documentation": "https://www.ricequant.com/doc/rqdata/python/",
                    "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sorted(destination.glob("*")) if path.suffix in {".feather", ".csv"}}}
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("历史 ST / 停牌 / 上退市状态已保存；原始行情未修改。", flush=True)
    finally:
        rqdatac.reset()
        if original is None:
            os.environ.pop("RQDATAC2_CONF", None)
        else:
            os.environ["RQDATAC2_CONF"] = original


def main():
    parser = argparse.ArgumentParser(description="下载本机私有历史市场状态，不保存或输出凭据")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--license-script", type=Path)
    arguments = parser.parse_args()
    try:
        fetch(arguments.data, arguments.output, arguments.start, arguments.end, arguments.license_script)
    except Exception as error:
        raise SystemExit(f"下载失败：{type(error).__name__}。请检查本机授权和连接；为保护凭据未输出服务器异常正文。") from None


if __name__ == "__main__":
    main()
