# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(layout='wide')

from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Grid, Funnel
from pyecharts.commons.utils import JsCode
from params import *
import copy
from utils import *
from scipy.stats import norm
import pandas as pd
from streamlit_echarts import st_pyecharts


np.seterr(divide='ignore', invalid='ignore')

st.header('ADRFI Toolkit')
st.text('')


def reset():
    with open('Start.py', 'a', encoding='utf-8') as f:
        f.write('#')
    st.experimental_rerun()


def main():
    global dps

    new_input = {}

    st.sidebar.markdown('## Set A DRFI Strategy:')

    for k in show_name:
        try:
            dvalue = dps[user_input][0][k] * 100 if k in input_pct else dps[user_input][0][k]
            nv = st.sidebar.text_input(show_name[k], dvalue, key=k)
            nv = float(nv) / 100.0 if k in input_pct else float(nv)
            new_input[k] = nv
        except:
            pass

    bg_params = st.sidebar.empty()
    bg_expander = bg_params.beta_expander('Other model settings')
    new_bg_input = {}
    for k in dps[bg_input][0]:
        new_bg_input[k] = float(bg_expander.text_input(k, value=dps[bg_input][0][k], key=k))

    st.sidebar.text('')
    st.sidebar.text('')
    if st.sidebar.button('Save this strategy'):
        dps[bg_input].append(new_bg_input)
        dps[user_input].append(new_input)

    if st.sidebar.button('reset the strategies'):
        reset()
        pass

    pses = []  # 计算过程的多组参数

    cb_AMTs = []
    for i in range(len(dps[user_input])):
        dps_ = copy.deepcopy(dps)
        uinp = copy.deepcopy(dps_[user_input][i])
        bginp = copy.deepcopy(dps_[bg_input][i])
        dps_.update(uinp)
        dps_.update(bginp)
        del dps_[user_input]
        bginp = copy.deepcopy(dps_[bg_input][i])
        dps_.update(bginp)
        del dps_[bg_input]

        dps_[cb_AMT] = dps_[cb_limit] - dps_[cb_att]
        cb_AMTs.append(dps_[cb_AMT])
        dps_[tier1_RL] = dps_[tier1_AMT] * dps_[tier1_BenRate]
        dps_[tier1_Int] = dps_[tier1_AMT] * dps_[tier1_IntRate]
        dps_[tier2_RL] = dps_[tier1_RL] * dps_[tier2_BenRate]
        dps_[tier2_Int] = dps_[tier2_RL] * dps_[tier2_IntRate]
        dps_[dataSR] = min(dps_[data_Invest] * dps_[dataSR_ImpRatio], dps_[dataSR_Cap])
        dps_[dataCBLoad] = dps_[dataCBLoad_Base] * max(0, 1 - dps_[data_Invest] * dps_[dataCBLoad_ImpRatio])
        dps_[ipSR] = min(dps_[ip_Invest] * dps_[ipSR_ImpRate], dps_[ipSR_Cap])
        dps_[ipCBLoad] = dps_[ipCBLoad_Base] * max(0, 1 - dps_[ip_Invest] * dps_[ipCBLoad_ImpRatio])
        dps_[ipRLLoad] = dps_[ipRLLoad_Base] * max(0, 1 - min(dps_[ip_Invest] * dps_[ipRLLoad_ImpRatio],
                                                              dps_[ipRLLoad_ImpRatio_Cap]))
        dps_[inSize_Base] = dps_[GDP] * dps_[ip_Base] * dps_[cap_Converter]
        # dps_[inSize] = dps_[inSize_Base] * (1 + min(dps_[ip_Invest] * dps_[inSize_ImpRatio], dps_[inSize_ImpRatio_Cap]))
        dps_[inSize] = dps_[GDP] * dps_[ip_Revised] * dps_[cap_Converter]

        dps_[unit] = round(3 * dps_[L200Y] / dps_[no_cells], 0)
        dps_[SharpeRatio] = dps_[SharpeRatioBaseline] * (1 - dps_[dataSR]) * (1 - dps_[ipSR])

        pses.append(dps_)
        pass

    AnInt = np.array(range(0, int(dps[no_cells]) + 1, 1))
    Unit = round(3 * dps[L200Y] / dps[no_cells], 0)
    LossAmountMillion: np.ndarray = AnInt * Unit
    pencentile = [0.75, 0.9, 0.95, 0.975, 0.99, 0.995]
    pctile_year = [4, 10, 20, 40, 100, 200]

    # x axis
    EP_X = np.zeros(LossAmountMillion.size)

    S2 = 1 / 2 * (0.01 + 0.05) * (dps[L100Y] - dps[L20Y])
    S3 = 1 / 2 * (0.01 + 0.005) * (dps[L200Y] - dps[L100Y])
    S3 = 1 / 2 * (0.01 + 0.005) * (dps[L200Y] - dps[L100Y])
    S4 = dps[PctAELTail] * dps[AEL]
    S1 = dps[AEL] - S2 - S3 - S4
    f1 = S1 - dps[L20Y] * 0.05
    f2 = f1 / (dps[L0Y] - 0.05)
    b = 1 / f2
    r = S4 / 0.005

    t1 = LossAmountMillion < dps[L20Y]
    t2 = LossAmountMillion >= dps[L20Y]
    t3 = LossAmountMillion < dps[L100Y]
    t4 = LossAmountMillion >= dps[L100Y]
    t5 = LossAmountMillion < dps[L200Y]
    t6 = LossAmountMillion >= dps[L200Y]
    EP_X[t1] = func1(LossAmountMillion[t1], b, dps[L0Y])
    EP_X[t2 & t3] = func2(LossAmountMillion[t2 & t3], dps[L20Y], dps[L100Y])
    EP_X[t4 & t5] = func3(LossAmountMillion[t4 & t5], dps[L100Y], dps[L200Y])
    EP_X[t6] = func4(LossAmountMillion[t6], r, dps[L200Y])
    CDF = 1 - EP_X

    st.markdown('*****')
    st.markdown('### I. Exceedance Probablility Curve')
    linedata = [e for e in zip(LossAmountMillion, EP_X)]
    xdata = [e[0] for e in linedata]
    ydata = [e[1] for e in linedata]
    c2 = (
        Line(init_opts=opts.InitOpts())
            .add_xaxis(xdata)
            .add_yaxis('Exceedance Probablility', ydata, is_smooth=True, is_symbol_show=False, symbol_size=0)
            .set_global_opts(tooltip_opts=opts.TooltipOpts(is_show=True, trigger='axis', axis_pointer_type='line'),
                             xaxis_opts=opts.AxisOpts(type_='value', name='Loss Amount'),
                             yaxis_opts=opts.AxisOpts(type_='value', name='Exceedance Probablility', is_scale=True),
                             datazoom_opts=opts.DataZoomOpts(is_show=True, type_='slider', range_start=0,
                                                             range_end=100),
                             title_opts=opts.TitleOpts(''))
    )
    st_pyecharts(c2, height='400%', width='100%')

    st.markdown('*****')
    st.markdown('### II. Loss by return year period ($M)')
    LossByReturnYearPeriod_empty = st.empty()

    st.markdown('*****')
    st.markdown('### III. Cost Comparison')
    CostComparison_empty = st.empty()

    st.markdown('****')
    st.markdown('### IV. Disaster Risk Layering')

    RetainedLossWithoutPolicy = LossAmountMillion / dps[natBdt]

    series_num = 1

    c = (
        Bar(init_opts=opts.InitOpts())
            .add_xaxis([str(y) + '-Year' for y in pctile_year])
            .add_yaxis('Without DRFI Strategy',
                       [round(RetainedLossWithoutPolicy[np.argmin(np.abs(CDF - pctile))] * 100, 2) for pctile in
                        pencentile], label_opts=opts.LabelOpts(is_show=True, formatter='{c}%'))
            .set_global_opts(xaxis_opts=opts.AxisOpts(type_='category', name='\n\nReturn\nYear\nPeriod'),
                             yaxis_opts=opts.AxisOpts(type_='value', name=' % of\nNational Budget', is_scale=True,
                                                      axislabel_opts=opts.LabelOpts(formatter=JsCode(
                                                          "function (x){return x + '%'}"
                                                      ))),
                             legend_opts=opts.LegendOpts(),
                             datazoom_opts=opts.DataZoomOpts(xaxis_index=[0, 1], type_='inside', range_start=0,
                                                             range_end=100, pos_bottom='0%'),
                             tooltip_opts=opts.TooltipOpts(is_show=True, trigger='axis',
                                                           axis_pointer_type='shadow',
                                                           is_show_content=True))
    )

    c1 = (
        Bar(init_opts=opts.InitOpts())
            .add_xaxis([str(y) + '-Year' for y in pctile_year])
            .add_yaxis('', [0] * len(pctile_year), label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_='category', name='', axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(type_='value', name='Financing\nUtilization', is_scale=True, is_inverse=True,
                                     axislabel_opts=opts.LabelOpts(formatter=JsCode(
                                         "function (x){return x + ' m'}"
                                     ))),
            legend_opts=opts.LegendOpts(pos_top='3%'),
            tooltip_opts=opts.TooltipOpts(is_show=True, is_show_content=True),
        )
    )
    c1_label = opts.LabelOpts(is_show=True,
                              position='inside',
                              color='white',
                              formatter=JsCode(
                                  "function(x) {d=x.data; if(d!==0.0){return d.toFixed() + ' m'}else{return ''};}"))

    for i in range(len(pses)):
        ps = pses[i]
        if i == len(pses) - 1:
            funnel_data = [
                [
                    f'\n\n\nLayer 1:\n\nContingent budget or reserve\n\n0~{int(ps[tier1_UL])} m\n\n\nHigh{" " * 150}Low\nFrequency{" " * 140}Severity',
                    30],
                [f'Layer 2:\n\nContingent credit\n\n{int(ps[tier1_UL])}~{int(ps[tier2_UL])} m', 20],
                [
                    f'Low Frequency{" " * 20}High Severity\n\n\n\nLayer 3:\n\nCapacity\nbuilding\nvehicles\n\n{int(ps[tier2_UL])}~{int(ps[tier3_UL])} m\n',
                    10]
            ]
            layerc = (Funnel(init_opts=opts.InitOpts(theme='dark'))
                      .add('layers', funnel_data, sort_='ascending',
                           label_opts=opts.LabelOpts(position='inside', color='black', font_weight=['bold']))
                      .set_global_opts(legend_opts=opts.LegendOpts(is_show=False),
                                       tooltip_opts=opts.TooltipOpts(is_show=False)))
            st_pyecharts(layerc, width='61.8%', height='600%')

        Limit = ps[inSize]

        RevisedLossRetainedRatio = ps[ipRLLoad]
        PctLossRetained = RevisedLossRetainedRatio
        LossTransfer = 1 - PctLossRetained

        InsurancePayout = (LossAmountMillion - ps[attachment]).clip(0, Limit) * LossTransfer

        CatbondSize = ps[cb_limit] - ps[cb_att]
        RetainedLossAfterInsurance = LossAmountMillion - InsurancePayout
        tier1_utilization = np.minimum(RetainedLossAfterInsurance, ps[tier1_AMT])
        Layer1FinancingBenefit = tier1_utilization * ps[tier1_BenRate]
        RetainedLossAfterLayer1Financing = np.maximum(0,
                                                      RetainedLossAfterInsurance - tier1_utilization - Layer1FinancingBenefit)
        tier2_utilization = np.minimum(RetainedLossAfterLayer1Financing, ps[tier2_AMT])
        Layer2FinancingBenefit = tier2_utilization * ps[tier2_BenRate]
        RetainedLossAfterLayer2Financing = np.maximum(0,
                                                      RetainedLossAfterLayer1Financing - tier2_utilization - Layer2FinancingBenefit)
        CatbondRecovery = (RetainedLossAfterLayer2Financing - ps[cb_att]).clip(0, CatbondSize)

        rl_pct_NB = (RetainedLossAfterLayer2Financing - CatbondRecovery) / ps[natBdt]

        layer1_insurance_payout = np.minimum(LossAmountMillion, ps[tier1_UL])
        layer2_insurance_payout = np.maximum(np.minimum(LossAmountMillion, ps[tier2_UL]) - ps[tier1_UL], 0)
        layer3_insurance_payout = np.maximum(np.minimum(LossAmountMillion, ps[tier3_UL]) - ps[tier2_UL], 0)

        WT_CDF = np.zeros(CDF.size)
        WT_CDF[CDF != 0] = norm.cdf(norm.ppf(CDF[CDF != 0]) - ps[SharpeRatio])

        TransformedProbability = np.diff(WT_CDF)
        TransformedProbability = np.insert(TransformedProbability, 0, WT_CDF[0])
        InsurancePremium = (TransformedProbability * InsurancePayout).sum()
        InsurancePremiumAsPctNationalBudget = InsurancePremium / ps[natBdt]

        tier1_wtCost = (TransformedProbability * layer1_insurance_payout).sum()
        tier1_wtCost /= float(ps[tier1_UL])
        tier2_wtCost = (TransformedProbability * layer2_insurance_payout).sum()
        tier2_wtCost /= (ps[tier2_UL] - ps[tier1_UL])
        tier3_wtCost = (TransformedProbability * layer3_insurance_payout).sum()
        tier3_wtCost /= (ps[tier3_UL] - ps[tier2_UL])

        cc_df1 = pd.DataFrame(
            [str(round(ps[tier1_IntRate] * 100, 0)) + '%', str(round(ps[tier2_IntRate] * 100, 0)) + '%'],
            index=['Debt', 'Credit Line'], columns=[''])

        cc_df2 = pd.DataFrame([str(round(tier1_wtCost * 100, 1)) + '%', str(round(tier2_wtCost * 100, 1)) + '%',
                               str(round(tier3_wtCost * 100, 1)) + '%'], index=[f'Layer {i}' for i in range(1, 4)],
                              columns=[''])
        sb, ertcc = CostComparison_empty.beta_columns(2)
        sb.markdown('&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Sovereign Borrowing**')
        ertcc.markdown('&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Estimated Risk Transfer Capacity Cost**')
        sb.table(cc_df1.T)
        ertcc.table(cc_df2.T)

        PDF = np.diff(CDF)
        PDF = np.insert(PDF, 0, CDF[0])
        ExpectedLoss = (CatbondRecovery * PDF).sum()
        CatbondAnnualCost = ExpectedLoss * ps[hisMul] * (1 + ps[ipCBLoad]) * (1 + ps[dataCBLoad])
        CatbondCoupon = CatbondAnnualCost
        CatbondCostAsPctNationalBudget = CatbondCoupon / ps[natBdt]

        InsurancePenetrationCostAsPctNationalBudget = ps[ip_Invest]
        DataInfrastractureCostAsPctNationalBudget = ps[data_Invest]

        Layer1BorrowingCost = ps[tier1_Int]
        Layer1CostAsPctNationalBudget = Layer1BorrowingCost / ps[natBdt]
        Layer2BorrowingCost = ps[tier2_Int]
        Layer2CostAsPctNationalBudget = Layer2BorrowingCost / ps[natBdt]

        TotalDRFIStrategyCostAsPctNatlBudget = InsurancePremiumAsPctNationalBudget + CatbondCostAsPctNationalBudget \
                                               + DataInfrastractureCostAsPctNationalBudget + InsurancePenetrationCostAsPctNationalBudget \
                                               + Layer1CostAsPctNationalBudget + Layer2CostAsPctNationalBudget

        VaR_pct_NationalBudget = rl_pct_NB + TotalDRFIStrategyCostAsPctNatlBudget

        serie_name = f'DRFI Strategy {n2c[series_num]}'
        serie_data = []
        LossByReturnYearPeriod = []
        l1FUs, l2Fus, cbrs = [], [], []
        for pctile in pencentile:
            position = np.argmin(np.abs(CDF - pctile))
            var = VaR_pct_NationalBudget[position]
            var = round(var * 100, 2)
            serie_data.append(var)
            loss = LossAmountMillion[position]
            LossByReturnYearPeriod.append(loss)
            l1FU = tier1_utilization[position]
            l1FU = round(l1FU, 0)
            l1FUs.append(l1FU)
            l2FU = tier2_utilization[position]
            l2FU = round(l2FU, 0)
            l2Fus.append(l2FU)
            cbr = CatbondRecovery[position]
            cbr = round(cbr, 0)
            cbrs.append(cbr)

        c.add_yaxis(serie_name, serie_data, label_opts=opts.LabelOpts(is_show=True, formatter='{c}%'))
        c1.add_yaxis('Layer 1 Financing Utilization', l1FUs, stack=serie_name, label_opts=c1_label) \
            .add_yaxis('Layer 2 Financing Utilization', l2Fus, stack=serie_name, label_opts=c1_label) \
            .add_yaxis('Capacity Building Vehicles Recovery', cbrs, stack=serie_name, label_opts=c1_label)

        LossByReturnYearPeriod_df = pd.DataFrame({'Loss ($m)': LossByReturnYearPeriod},
                                                 index=[str(y) + '-year' for y in pctile_year]).T
        LossByReturnYearPeriod_empty.table(LossByReturnYearPeriod_df)

        series_num += 1

    grid = (
        Grid()
            .add(c, grid_opts=opts.GridOpts(pos_top='15%', pos_left='8%', height='43%', width='79%'), grid_index=0)
            .add(c1, grid_opts=opts.GridOpts(pos_bottom='8%', pos_left='8%', height='30%', width='79%'), grid_index=1)
    )

    df = pd.DataFrame(dps[user_input])
    df[cb_AMT] = cb_AMTs
    params_df = df.copy(deep=True)

    rn = copy.deepcopy(show_name)
    for pname in change_to_amt:
        params_df[pname] = params_df[pname].map(lambda x: str(round(x * dps[natBdt], 1))[:3])

    for pname in pct_params:
        params_df[pname] = params_df[pname].map(lambda x: str(round(x * 100, 2)) + '%')
    rn.update(bg_show_name)
    params_df.rename(columns=rn, inplace=True)
    params_df = params_df.loc[:, [show_name[k] for k in show_name]]
    name_change = {}
    for pname in change_to_amt:
        if pname in show_name:
            name_change[show_name[pname]] = show_name[pname].replace(' as % National Budget', '')
    params_df.rename(columns=name_change, inplace=True)
    params_df = pd.DataFrame(params_df.values.T, index=params_df.columns,
                             columns=[f'DRFI Strategy {n2c[i + 1]}' for i in params_df.index])
    st.markdown('*****')
    st.markdown('### V. The DRFI Strategies')
    st.dataframe(params_df)

    st.markdown('*****')
    st.markdown('### VI. Loss Impact as % of National Budget under Various Scenarios')
    st_pyecharts(grid, width='100%', height='618%', renderer='canvas')


main()
#