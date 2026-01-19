import akshare as ak
import pandas as pd
import os

OUTPUT_PATH = './data/raw/FNSPID/SP500_Index.csv'

def download_sp500_fixed():
    print(">>> 正在尝试使用 AkShare (修正版) 获取 S&P 500 数据...")
    try:
        # 【修正点】: 函数名变了，现在用这个
        df = ak.index_us_stock_sina(symbol=".INX")
        
        print("获取成功，开始清洗...")
        # 打印一下列名，防止出错
        print("原始列名:", df.columns.tolist())
        
        # AkShare 返回的列名通常是中文的，如 "日期", "收盘价" 等，或者是 date, close
        # 我们做一个自动映射，不管它返回中文还是英文都能处理
        rename_map = {
            'date': 'Date', '日期': 'Date',
            'close': 'Market_Close', '收盘价': 'Market_Close',
            'volume': 'Market_Vol', '成交量': 'Market_Vol'
        }
        df = df.rename(columns=rename_map)
        
        # 确保是时间格式
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 筛选 2019-2023
        mask = (df['Date'] >= '2019-01-01') & (df['Date'] <= '2023-12-31')
        df = df.loc[mask].sort_values('Date')
        
        # 只保留需要的列
        df = df[['Date', 'Market_Close', 'Market_Vol']]
        
        # 确保目录存在
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ 下载成功！已保存至: {OUTPUT_PATH}")
        print(df.head())
        
    except AttributeError:
        print("❌ 依然报错？看来你的 AkShare 版本比较特殊。建议直接使用【方案二】手动下载，一分钟搞定。")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

if __name__ == "__main__":
    download_sp500_fixed()