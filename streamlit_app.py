import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title='Comex Asset Optimization — Basic', page_icon='🏭', layout='wide')

# Public PPG data, USD millions. Comex is not separately reported as a full
# standalone balance sheet; Global Architectural Coatings is used as the
# operating proxy and PPG consolidated working-capital data as benchmark.
DATA = {
2025: dict(segment_assets=6676, segment_sales=3838, segment_income=599, segment_ebitda=708,
          latam_sales=1314, cash=2163, sti=56, receivables=3336, inventory=1996,
          other_current=408, current_assets=7959, ppe=4005, goodwill=6149,
          intangibles=1971, deferred_tax=481, investments=332, lease_assets=604,
          other_assets=597, total_assets=22098, current_liabilities=4900,
          long_debt=6602, trade_receivables=2783, inventory_fifo=2177,
          trade_liabilities=2212, dso=59, inv_turnover=4.4, owc=2748,
          owc_pct=0.176, capex=778, ocf=1936),
2024: dict(segment_assets=5887, segment_sales=3921, segment_income=678, segment_ebitda=782,
          latam_sales=1332, cash=1270, sti=88, receivables=2985, inventory=1846,
          other_current=368, current_assets=6557, ppe=3464, goodwill=5690,
          intangibles=1922, deferred_tax=303, investments=331, lease_assets=597,
          other_assets=599, total_assets=19433, current_liabilities=5014,
          long_debt=4876, trade_receivables=2477, inventory_fifo=2015,
          trade_liabilities=2161, dso=51, inv_turnover=4.5, owc=2331,
          owc_pct=0.156, capex=721, ocf=1391)
}

SOURCES = [
'https://www.sec.gov/Archives/edgar/data/79879/000007987926000090/ppg2025annualreport.pdf',
'https://www.sec.gov/Archives/edgar/data/79879/000007987926000046/ppg-20251231.htm',
'https://investor.ppg.com/news/news-details/2026/PPG-reports-fourth-quarter-and-full-year-2025-financial-results/default.aspx'
]

def ratios(d):
    return pd.Series({
        'Current ratio': d['current_assets']/d['current_liabilities'],
        'Quick ratio': (d['cash']+d['sti']+d['receivables'])/d['current_liabilities'],
        'Asset turnover': d['segment_sales']/d['segment_assets'],
        'Segment operating margin': d['segment_income']/d['segment_sales'],
        'Segment EBITDA margin': d['segment_ebitda']/d['segment_sales'],
        'Trade receivables / sales': d['trade_receivables']/d['segment_sales'],
        'Inventory / sales': d['inventory_fifo']/d['segment_sales'],
        'DSO (days)': d['trade_receivables']/d['segment_sales']*365,
        'Inventory turnover': d['inv_turnover'],
        'Operating working capital / sales': d['owc']/d['segment_sales'],
        'PPE / total assets': d['ppe']/d['total_assets']})

def optimize_wc(sales, cash, ar, inv, min_cash_ratio, target_dso, target_dio):
    min_cash = min_cash_ratio * sales
    target_ar = sales * target_dso / 365
    target_inv = sales * 0.60 * target_dio / 365
    x0 = np.array([max(cash,min_cash), ar, inv], dtype=float)
    bounds=[(min_cash, max(cash*1.5,sales*.1)), (sales*.01, ar*1.25), (sales*.01, inv*1.25)]
    cons=[{'type':'ineq','fun':lambda x: target_dso-x[1]/sales*365},
          {'type':'ineq','fun':lambda x: target_dio-x[2]/(sales*.60)*365}]
    res=minimize(lambda x:x.sum(),x0,method='SLSQP',bounds=bounds,constraints=cons,
                 options={'maxiter':1000,'ftol':1e-9})
    return pd.Series(res.x if res.success else [min_cash,target_ar,target_inv],index=['Cash','Receivables','Inventory'])

st.title('Comex — Basic Asset Optimization')
st.caption('Academic basic-level model using public PPG financial information.')
st.warning('Comex is a PPG brand/business. PPG does not publish a complete standalone Comex balance sheet in its public 2025 Form 10-K. This app therefore uses Global Architectural Coatings as the operating proxy for Comex and PPG consolidated working-capital data as a benchmark. It does not represent PPG consolidated assets as Comex-only assets.')

with st.sidebar:
    st.header('Model variables')
    year=st.selectbox('Financial year',[2025,2024],index=0)
    objective=st.selectbox('Optimization objective',['Minimize working-capital assets','Maintain current structure'],index=0)
    min_cash_ratio=st.slider('Minimum cash / annual sales',0.01,0.30,0.10,0.01)
    target_dso=st.slider('Target DSO (days)',20,90,50,1)
    target_dio=st.slider('Target inventory days',20,150,60,1)
    run=st.button('Run Asset Optimization',type='primary',use_container_width=True)

d=DATA[year]

st.subheader('1. Financial data used')
a,b,c,e=st.columns(4)
a.metric('Segment assets',f"${d['segment_assets']:,.0f} M")
b.metric('Segment sales',f"${d['segment_sales']:,.0f} M")
c.metric('Segment income',f"${d['segment_income']:,.0f} M")
e.metric('Segment EBITDA',f"${d['segment_ebitda']:,.0f} M")
st.write(f"**Global Architectural Coatings — Latin America sales:** ${d['latam_sales']:,.0f} million.")

fin=pd.DataFrame({'Item':['Cash','Short-term investments','Receivables','Inventories','Other current assets','Total current assets','PP&E, net','Goodwill','Intangibles','Total assets','Current liabilities','Long-term debt'],
'USD millions':[d['cash'],d['sti'],d['receivables'],d['inventory'],d['other_current'],d['current_assets'],d['ppe'],d['goodwill'],d['intangibles'],d['total_assets'],d['current_liabilities'],d['long_debt']]})
st.dataframe(fin,hide_index=True,use_container_width=True)

st.subheader('2. Basic financial ratios')
r=ratios(d)
rd=pd.DataFrame({'Metric':r.index,'Value':r.values})
st.dataframe(rd.style.format({'Value':'{:.2f}'}),hide_index=True,use_container_width=True)

st.subheader('3. Asset structure benchmark')
asset_mix=pd.Series({'Cash + short-term investments':d['cash']+d['sti'],'Receivables':d['receivables'],'Inventories':d['inventory'],'Other current assets':d['other_current'],'PP&E':d['ppe'],'Goodwill':d['goodwill'],'Intangibles':d['intangibles'],'Other non-current assets':d['deferred_tax']+d['investments']+d['lease_assets']+d['other_assets']})
mix=pd.DataFrame({'Asset category':asset_mix.index,'USD millions':asset_mix.values,'% of total assets':asset_mix.values/d['total_assets']})
st.dataframe(mix.style.format({'USD millions':'${:,.0f}','% of total assets':'{:.2%}'}),hide_index=True,use_container_width=True)
fig,ax=plt.subplots(figsize=(9,5)); ax.bar(mix['Asset category'],mix['% of total assets']); ax.set_ylabel('% of total assets'); ax.set_title(f'PPG consolidated asset mix benchmark — {year}'); ax.tick_params(axis='x',rotation=45); ax.grid(axis='y',alpha=.25); st.pyplot(fig,clear_figure=True)

if run:
    current=pd.Series({'Cash':d['cash'],'Receivables':d['trade_receivables'],'Inventory':d['inventory_fifo']})
    if objective=='Maintain current structure': optimized=current.copy()
    else: optimized=optimize_wc(d['segment_sales'],d['cash'],d['trade_receivables'],d['inventory_fifo'],min_cash_ratio,target_dso,target_dio)
    comp=pd.DataFrame({'Current':current,'Optimized':optimized}); comp['Change']=comp['Optimized']-comp['Current']; comp['Change %']=comp['Change']/comp['Current']
    st.subheader('4. Current vs. optimized operating assets')
    st.dataframe(comp.style.format({'Current':'${:,.0f}','Optimized':'${:,.0f}','Change':'${:,.0f}','Change %':'{:.2%}'}),use_container_width=True)
    cur=current.sum(); opt=optimized.sum(); release=cur-opt
    x,y,z=st.columns(3); x.metric('Current operating assets',f'${cur:,.0f} M'); y.metric('Optimized operating assets',f'${opt:,.0f} M'); z.metric('Potential capital released',f'${release:,.0f} M')
    st.subheader('5. Optimization constraints')
    st.write(f'- Minimum cash / sales: **{min_cash_ratio:.1%}**')
    st.write(f'- Target DSO: **{target_dso} days**')
    st.write(f'- Target inventory days: **{target_dio} days**')
    st.subheader('6. Basic interpretation')
    st.write('The optimizer minimizes capital tied up in cash, receivables and inventory subject to the selected liquidity and operating-efficiency constraints. This is a basic working-capital optimization, not a valuation model.')

st.subheader('7. Sources and methodological note')
st.write('PPG identifies COMEX as a primary brand within Architectural Coatings Latin America and Asia Pacific. The 2025 filing reports Global Architectural Coatings segment assets of $6.676 billion, segment sales of $3.838 billion, and Latin America sales of $1.314 billion. The same filing reports audited consolidated balance-sheet and working-capital information for PPG.')
st.write('PPG also lists Mexican Comex subsidiaries including Comercial Mexicana de Pinturas, Comex Industrial Coatings, Consorcio Comex and Grupo Comex as wholly owned subsidiaries. Because a complete standalone Comex balance sheet is not disclosed publicly in the filing, the proxy is explicitly identified.')
for s in SOURCES: st.markdown(f'- {s}')
st.caption('USD millions unless stated otherwise. Academic basic-level model; not an audited standalone Comex financial statement.')
