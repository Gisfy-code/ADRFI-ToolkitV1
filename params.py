# -*-coding: utf-8 -*-
import copy

tier1_UL = 'tier1_UL'
tier2_UL = 'tier2_UL'
tier3_UL = 'tier3_UL'
cb_att = 'cb_att'
cb_limit = 'cb_limit'

tier1_AMT = 'tier1_AMT'
tier2_AMT = 'tier2_AMT'
cb_AMT = 'cb_AMT'

tier1_BenRate = 'tier1_BenRate'
tier1_RL = 'tier1_RL'
tier1_IntRate = 'tier1_IntRate'
tier1_Int = 'tier1_Int'

tier2_BenRate = 'tier2_BenRate'
tier2_RL = 'tier2_RL'
tier2_IntRate = 'tier2_IntRate'
tier2_Int = 'tier2_Int'

data_Invest = 'data_Invest'
dataSR_ImpRatio = 'dataSR_ImpRatio'
dataSR = 'dataSR'
dataSR_Cap = 'dataSR_Cap'

dataCBLoad_Base = 'dataCBLoad_Base'
dataCBLoad_ImpRatio = 'dataCBLoad_ImpRatio'
dataCBLoad = 'dataCBLoad'

ip_Invest = 'ip_Invest'
ipSR_ImpRate = 'ipSR_ImpRate'
ipSR = 'ipSR'
ipSR_Cap = 'ipSR_Cap'

ipCBLoad_Base = 'ipCBLoad_Base'
ipCBLoad_ImpRatio = 'ipCBLoad_ImpRatio'
ipCBLoad = 'ipCBLoad'

ipRLLoad_Base = 'ipRLLoad_Base'
ipRLLoad_ImpRatio = 'ipRLLoad_ImpRatio'
ipRLLoad = 'ipRLLoad'
ipRLLoad_ImpRatio_Cap = 'ipRLLoad_ImpRatio_Cap'

ip_Base = 'ip_Base'
cap_Converter = 'cap_Converter'
ip_ImpRatio = 'ip_ImpRatio'
ip_Revised = 'ip_Revised'

inSize_Base = 'inSize_Base'
inSize_ImpRatio = 'inSize_ImpRatio'
inSize = 'inSize'
inSize_ImpRatio_Cap = 'inSize_ImpRatio_Cap'

AEL = 'AEL'
PctAELTail = 'PctAELTail'
L200Y = 'L200Y'
L100Y = 'L100Y'
L20Y = 'L20Y'
L0Y = 'L0Y'
SharpeRatioBaseline = 'SharpeRatioBaseline'
SharpeRatio = 'SharpeRatio'
no_cells = 'no_cells'
unit = 'unit'
hisMul = 'hisMul'

natBdt = 'natBdt'  # National Budget

attachment = 'attachment'

user_input = 'user_input'
bg_input = 'bg_input'

GDP = 'GDP'

show_name = {
    tier1_UL: 'Layer 1: Low Risk Layer ',
    tier2_UL: 'Layer 2: Medium Risk Layer ',
    tier3_UL: 'Layer 3: High Risk Layer',

    tier1_AMT: 'Layer 1 Financing Amount',
    tier2_AMT: 'Layer 2 Financing Amount',

    cb_att: 'Capacity Building Vehicles Attachment',
    cb_limit: 'Capacity Building Vehicles Limit',
    cb_AMT: 'Capacity Building Vehicles Amount',

    data_Invest: 'Investment on Data Infrastructure as % National Budget',

    ip_Invest: 'Investment to improve insurance penetration as % National Budget',
}

bg_show_name = {
}

pct_params = []
change_to_amt = [data_Invest, ip_Invest]

dps = {
    user_input: [{
        tier1_UL: 50.0,
        tier1_AMT: 10.0,
        tier2_UL: 100.0,
        tier2_AMT: 10.0,
        tier3_UL: 600.0,
        cb_att: 200.00,
        cb_limit: 250.0,
        data_Invest: 0.001,
        ip_Invest: 0.001,

    }],

    bg_input: [{
        SharpeRatioBaseline: 0.6,

        tier1_BenRate: 0.5,
        tier1_IntRate: 0.08,

        tier2_BenRate: 0.4,
        tier2_IntRate: 0.02,

        dataSR_ImpRatio: 100.0,
        dataSR_Cap: 0.4,

        dataCBLoad_Base: 0.5,
        dataCBLoad_ImpRatio: 200.0,

        ipSR_ImpRate: 200.0,
        ipSR_Cap: 0.4,

        ipCBLoad_Base: 0.4,
        ipCBLoad_ImpRatio: 200.0,

        ipRLLoad_Base: 0.5,
        ipRLLoad_ImpRatio: 300.0,
        ipRLLoad_ImpRatio_Cap: 0.6,

        ip_Base: 0.005,
        cap_Converter: 4.0,
        ip_ImpRatio: 0.1,
        ip_Revised: 0.006,

        # inSize_Base: 200.0,
        inSize_ImpRatio: 300.0,
        inSize_ImpRatio_Cap: 1.0,

        GDP: 18173.8,

        hisMul: 2.98,

        attachment: 0.0,
    }],

    L200Y: 1020.6,
    L100Y: 875.3,
    L20Y: 342.6,
    L0Y: 0.3,
    AEL: 52.3,
    PctAELTail: 0.02,

    natBdt: 1452.0,

    no_cells: 500.0,

}

dps_original = copy.deepcopy(dps)

input_pct = [data_Invest, ip_Invest]
