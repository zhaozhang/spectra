import json
import glob
import sys

#files = glob.glob("sne-2015-2019-master/SN2018feb.json")
files = glob.glob("sne-*-master/*.json")

type_dict = {
    "Ia": "Ia",
    "Ia?": "Ia",
    "Ia Pec": "Ia",
    "II P": "IIP",
    "~": "N/A",
    "Candidate": "N/A",
    "CC": "N/A",
    "Computed-IIb": "IIb",
    "Computed-PISN": "N/A",
    "I": "Ib/c",
    ".Ia": "Ia",
    "Ia-02cx": "Ia",
    "Ia-02ic-like": "Ia",
    "Ia-91bg": "Ia",
    "Ia-91T": "Ia",
    "Ia/b": "Ia",
    "Ia CSM": "Ia",
    "Ia-CSM/IIn": "Ia",
    "Ia-p": "Ia",
    "Iax[02cx-like]": "Ia",
    "Ib": "Ib",
    "Ib/c": "Ib",
    "Ib/c?": "Ib",
    "Ib/c-BL": "Ib",
    "Ib/IIb": "Ib",
    "Ibn": "Ib",
    "Ib Pec": "Ib",
    "Ic": "Ic",
    "Ic?": "Ic",
    "Ic BL": "Ic",
    "Ic Pec": "Ic",
    "II": "II",
    ".II": "II",
    "II?": "II",
    "II-09ip": "II",
    "IIb": "IIb",
    "IIb?": "IIb",
    "IIb/Ib/Ic": "IIb",
    "II L": "IIL",
    "IIn": "IIn",
    "IIn/LBV": "IIn",
    "IIn Pec": "IIn",
    "II Pec": "II",
    "I Pec": "I",
    "Iz": "Iz",
    "LBV/IIn": "LBV",
    "LBV to IIn": "LBV",
    "other": "N/A",
    "Other": "N/A",
    "SLSN": "I",
    "SLSN-I": "I",
    "SLSN-II": "II",
    "SNSN-II": "II",
    "TDE": "N/A",
    "Afterglow": "N/A",
    "AGN": "N/A",
    "BL-Ic": "Ic",
    "Ca-rich": "N/A",
    "Computed-Ia": "Ia",
    "Computed-IIP": "IIP",
    "Galaxy": "N/A",
    "Ia-99aa": "Ia",
    "Ia/c": "Ia",
    "Ia-HV": "Ia",
    ".IaPec": "Ia",
    "Ib-Ca": "Ib",
    "Ib (Ca rich)": "Ib",
    "Ib/Ic": "Ib",
    "Ib/Ic (Ca rich?)?": "Ib",
    "Ib/Ic (Ca rich)": "Ib",
    "Ib-IIb": "Ib",
    "Ibn/IIbn": "Ibn",
    "Ic/Ic-BL": "Ic",
    "I-faint": "N/A",
    "IIb/Ib": "IIb",
    "IIb/Ib/Ic (Ca rich)": "IIb",
    "IIn?": "IIn",
    "IIn/Ibn": "IIn",
    "IIn-pec/LBV": "IIn",
    ".IIP": "IIP",
    "II P?": "IIP",
    "II Pec?": "II",
    "II P Pec": "IIP",
    "LBV": "N/A",
    "Lensed SN Ia": "Ia",
    "PISN?": "N/A",
    "SLSN-I-R": "N/A",
    "SLSN-R": "N/A",
    "Super-Luminous Ic": "Ic",
    "Variable": "N/A"
}

label_dict = {
    "Ia": 0,
    "Ib": 1,
    "Ic": 1,
    "IIP": 1,
    "IIn": 1
}

selected = ["Ia", "Ib", "Ic", "IIP", "IIn"]

count = 0
f_out = open("id-source.csv", "w")
Ia_out = open("Ia.csv", "w")
Ib_out = open("Ib.csv", "w")
Ic_out = open("Ic.csv", "w")
IIP_out = open("IIP.csv", "w")
IIn_out = open("IIn.csv", "w")

length_list = []
for f in files:
    with open(f) as f_in:
        d = json.load(f_in)

    for k in d.keys():
        if 'spectra' in d[k]:
            if 'claimedtype' not in d[k]:
                break
            types = d[k]['claimedtype']


            if type_dict[types[0]['value']] in selected:            
                for s in d[k]['spectra']:
                    spectra = []
                    if 'data' in s:
                        for p in s['data']:
                            f_p = float(p[1])
                            spectra.append(f_p)
                        if len(spectra) < 2 or len(spectra) > 8192:
                            pass

                        spectra.extend([0]*(8192-len(spectra)))
                        if type_dict[types[0]['value']] == "Ia":
                            out = Ia_out
                        elif type_dict[types[0]['value']] == "Ib":
                            out = Ib_out
                        elif type_dict[types[0]['value']] == "Ic":
                            out = Ic_out
                        elif type_dict[types[0]['value']] == "IIP":
                            out = IIP_out
                        elif type_dict[types[0]['value']] == "IIn":
                            out = IIn_out
                        else:
                            print(f"error type ${type_dict[types[0]['value']]} not supported")
                            sys.exit()

                        out.write(f"{count}, {spectra}, {label_dict[type_dict[types[0]['value']]]}\n")
                        f_out.write(f"{count}, {type_dict[types[0]['value']]}, {label_dict[type_dict[types[0]['value']]]}, {k}\n")
                        count += 1

f_out.close()
Ia_out.close()
Ib_out.close()
Ic_out.close()
IIP_out.close()
IIn_out.close()
