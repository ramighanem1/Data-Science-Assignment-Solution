from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel


class ModelParams(BaseModel):
    P1:int
    P2:int
    P3:int
    P4:int
    P5:int
    P6:int
    P7:int
    P8:int
    P9:int
    P10:int
    P11:int
    P12:int
    P13:int
    P14:int
    P15:int
    P16:int
    P17:int
    P18:int
    P19:int
    P20:int
    P21:int
    P22:int
    P23:int
    P24:int
    P25:int
    P26:int
    P27:int
    P28:int
    P29:int
    P30:int
    P31:int
    P32:int
    P33:int
    P34:int
    P35:int
    P36:int
    P37:int
    P38:int
    P39:int
    P40:int
    P41:int
    P42:int
    P43:int
    P44:int
    P45:int
    P46:int
    P47:int
    P48:int
    P49:int
    P50:int
    P51:int
    P52:int
    P53:int
    P54:int
    P55:int
    P56:int
    P57:int
    P58:int
    P59:int
    P60:int
    P61:int
    P62:int
    P63:int
    P64:int
    P65:int
    P66:int
    P67:int
    P68:int
    P69:int
    P70:int
    P71:int
    P72:int
    P73:int
    P74:int
    P75:int
    P76:int
    P77:int
    P78:int
    P79:int
    P80:int
    P81:int
    P82:int
    P83:int
    P84:int
    P85:int
    P86:int
    P87:int
    P88:int
    P89:int
    P90:int
    P91:int
    P92:int
    P93:int
    P94:int
    P95:int
    P96:int
    P97:int
    P98:int
    P99:int
    P100:int
    P101:int
    P102:int
    P103:int
    P104:int
    P105:int
    P106:int
    P107:int
    P108:int
    P109:int
    P110:int
    P111:int
    P112:int
    P113:int
    P114:int
    P115:int
    P116:int
    P117:int
    P118:int
    P119:int
    P120:int
    P121:int
    P122:int
    P123:int
    P124:int
    P125:int
    P126:int
    P127:int
    P128:int
    P129:int
    P130:int
    P131:int
    P132:int



app = FastAPI()

clf = load('/model/my_model.joblib')

def get_prediction(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, P47, P48, P49, P50, P51, P52, P53, P54, P55, P56, P57, P58, P59, P60, P61, P62, P63, P64, P65, P66, P67, P68, P69, P70, P71, P72, P73, P74, P75, P76, P77, P78, P79, P80, P81, P82, P83, P84, P85, P86, P87, P88, P89, P90, P91, P92, P93, P94, P95, P96, P97, P98, P99, P100, P101, P102, P103, P104, P105, P106, P107, P108, P109, P110, P111, P112, P113, P114, P115, P116, P117, P118, P119, P120, P121, P122, P123, P124, P125, P126, P127, P128, P129, P130, P131, P132):
    x = [[P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, P47, P48, P49, P50, P51, P52, P53, P54, P55, P56, P57, P58, P59, P60, P61, P62, P63, P64, P65, P66, P67, P68, P69, P70, P71, P72, P73, P74, P75, P76, P77, P78, P79, P80, P81, P82, P83, P84, P85, P86, P87, P88, P89, P90, P91, P92, P93, P94, P95, P96, P97, P98, P99, P100, P101, P102, P103, P104, P105, P106, P107, P108, P109, P110, P111, P112, P113, P114, P115, P116, P117, P118, P119, P120, P121, P122, P123, P124, P125, P126, P127, P128, P129, P130, P131, P132]]
    y = clf.predict(x)[0]  
    prob = clf.predict_proba(x)[0].tolist()  
    classes = ['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne','Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma','Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis','Common Cold', 'Dengue', 'Diabetes ','Dimorphic hemmorhoids(piles)', 'Drug Reaction','Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack','Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E','Hypertension ', 'Hyperthyroidism', 'Hypoglycemia','Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine','Osteoarthristis', 'Paralysis (brain hemorrhage)','Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis','Typhoid', 'Urinary tract infection', 'Varicose veins','hepatitis A']
    return {'prediction': classes[int(y)], 'probability': prob}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{P1}/{P2}/{P3}/{P4}/{P5}/{P6}/{P7}/{P8}/{P9}/{P10}/{P11}/{P12}/{P13}/{P14}/{P15}/{P16}/{P17}/{P18}/{P19}/{P20}/{P21}/{P22}/{P23}/{P24}/{P25}/{P26}/{P27}/{P28}/{P29}/{P30}/{P31}/{P32}/{P33}/{P34}/{P35}/{P36}/{P37}/{P38}/{P39}/{P40}/{P41}/{P42}/{P43}/{P44}/{P45}/{P46}/{P47}/{P48}/{P49}/{P50}/{P51}/{P52}/{P53}/{P54}/{P55}/{P56}/{P57}/{P58}/{P59}/{P60}/{P61}/{P62}/{P63}/{P64}/{P65}/{P66}/{P67}/{P68}/{P69}/{P70}/{P71}/{P72}/{P73}/{P74}/{P75}/{P76}/{P77}/{P78}/{P79}/{P80}/{P81}/{P82}/{P83}/{P84}/{P85}/{P86}/{P87}/{P88}/{P89}/{P90}/{P91}/{P92}/{P93}/{P94}/{P95}/{P96}/{P97}/{P98}/{P99}/{P100}/{P101}/{P102}/{P103}/{P104}/{P105}/{P106}/{P107}/{P108}/{P109}/{P110}/{P111}/{P112}/{P113}/{P114}/{P115}/{P116}/{P117}/{P118}/{P119}/{P120}/{P121}/{P122}/{P123}/{P124}/{P125}/{P126}/{P127}/{P128}/{P129}/{P130}/{P131}/{P132}")
def predict(P1:int, P2:int, P3:int, P4:int, P5:int, P6:int, P7:int, P8:int, P9:int, P10:int, P11:int, P12:int, P13:int, P14:int, P15:int, P16:int, P17:int, P18:int, P19:int, P20:int, P21:int, P22:int, P23:int, P24:int, P25:int, P26:int, P27:int, P28:int, P29:int, P30:int, P31:int, P32:int, P33:int, P34:int, P35:int, P36:int, P37:int, P38:int, P39:int, P40:int, P41:int, P42:int, P43:int, P44:int, P45:int, P46:int, P47:int, P48:int, P49:int, P50:int, P51:int, P52:int, P53:int, P54:int, P55:int, P56:int, P57:int, P58:int, P59:int, P60:int, P61:int, P62:int, P63:int, P64:int, P65:int, P66:int, P67:int, P68:int, P69:int, P70:int, P71:int, P72:int, P73:int, P74:int, P75:int, P76:int, P77:int, P78:int, P79:int, P80:int, P81:int, P82:int, P83:int, P84:int, P85:int, P86:int, P87:int, P88:int, P89:int, P90:int, P91:int, P92:int, P93:int, P94:int, P95:int, P96:int, P97:int, P98:int, P99:int, P100:int, P101:int, P102:int, P103:int, P104:int, P105:int, P106:int, P107:int, P108:int, P109:int, P110:int, P111:int, P112:int, P113:int, P114:int, P115:int, P116:int, P117:int, P118:int, P119:int, P120:int, P121:int, P122:int, P123:int, P124:int, P125:int, P126:int, P127:int, P128:int, P129:int, P130:int, P131:int, P132:int):
    pred = get_prediction(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20, P21, P22, P23, P24, P25, P26, P27, P28, P29, P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P40, P41, P42, P43, P44, P45, P46, P47, P48, P49, P50, P51, P52, P53, P54, P55, P56, P57, P58, P59, P60, P61, P62, P63, P64, P65, P66, P67, P68, P69, P70, P71, P72, P73, P74, P75, P76, P77, P78, P79, P80, P81, P82, P83, P84, P85, P86, P87, P88, P89, P90, P91, P92, P93, P94, P95, P96, P97, P98, P99, P100, P101, P102, P103, P104, P105, P106, P107, P108, P109, P110, P111, P112, P113, P114, P115, P116, P117, P118, P119, P120, P121, P122, P123, P124, P125, P126, P127, P128, P129, P130, P131, P132)
    return pred


@app.post("/predict-post/")
def post_predict(params: ModelParams):
    pred = get_prediction(params.P1,params.P2,params.P3,params.P4,params.P5,params.P6,params.P7,params.P8,params.P9,params.P10,params.P11,params.P12,params.P13,params.P14,params.P15,params.P16,params.P17,params.P18,params.P19,params.P20,params.P21,params.P22,params.P23,params.P24,params.P25,params.P26,params.P27,params.P28,params.P29,params.P30,params.P31,params.P32,params.P33,params.P34,params.P35,params.P36,params.P37,params.P38,params.P39,params.P40,params.P41,params.P42,params.P43,params.P44,params.P45,params.P46,params.P47,params.P48,params.P49,params.P50,params.P51,params.P52,params.P53,params.P54,params.P55,params.P56,params.P57,params.P58,params.P59,params.P60,params.P61,params.P62,params.P63,params.P64,params.P65,params.P66,params.P67,params.P68,params.P69,params.P70,params.P71,params.P72,params.P73,params.P74,params.P75,params.P76,params.P77,params.P78,params.P79,params.P80,params.P81,params.P82,params.P83,params.P84,params.P85,params.P86,params.P87,params.P88,params.P89,params.P90,params.P91,params.P92,params.P93,params.P94,params.P95,params.P96,params.P97,params.P98,params.P99,params.P100,params.P101,params.P102,params.P103,params.P104,params.P105,params.P106,params.P107,params.P108,params.P109,params.P110,params.P111,params.P112,params.P113,params.P114,params.P115,params.P116,params.P117,params.P118,params.P119,params.P120,params.P121,params.P122,params.P123,params.P124,params.P125,params.P126,params.P127,params.P128,params.P129,params.P130,params.P131,params.P132)
    return pred