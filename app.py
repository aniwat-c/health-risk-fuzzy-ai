import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. การตั้งค่าหน้าเว็บและสไตล์ CSS แบบ Dark Mode ถาวร ---
st.set_page_config(page_title="AI Health Advisor Pro", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    /* 1. บังคับพื้นหลังแอปดำถาวร */
    .stApp {
        background-color: #0E1117 !important;
    }

    /* 2. หัวข้อหลักและ Label ทั่วไปเป็นสีขาว */
    h1, h2, h3, h4, label, span { 
        color: #FFFFFF !important; 
    }

    /* 3. จัดการ Slider (ลบกรอบสีแดงและติ่ง Tooltip ออกทั้งหมด) */
    
    /* บังคับสีของแถบ Slider หลัก */
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background-color: #FF4B4B !important;
    }

    /* ลบกรอบสีแดง (Focus Ring) และเงาเวลาคลิกหรือเลื่อน */
    .stSlider [data-baseweb="slider"] > div:focus,
    .stSlider [role="slider"]:focus,
    .stSlider [role="slider"]:active {
        outline: none !important;
        box-shadow: none !important;
    }

    /* ซ่อนติ่งสี่เหลี่ยมสีแดง (Tooltip) ที่เด้งขึ้นมาเวลาเลื่อน */
    div[data-baseweb="tooltip"] {
        display: none !important;
    }

    /* ปรับแต่งปุ่มวงกลม (Thumb) ให้เรียบร้อย */
    .stSlider [role="slider"] {
        background-color: #FF4B4B !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: none !important;
    }

    /* บังคับตัวเลขที่หัว Slider และสเกลตัวเลขให้เป็นสีขาวชัดเจน */
    .stSlider [data-testid="stTickBarMax"], 
    .stSlider [data-testid="stTickBarMin"],
    .stSlider div[data-baseweb="typography"],
    .stSlider div {
        color: #FFFFFF !important;
    }

    /* 4. Metric พื้นขาว ตัวหนังสือดำ */
    [data-testid="stMetricValue"] { color: #000000 !important; }
    [data-testid="stMetricLabel"] { color: #000000 !important; }
    .stMetric {
        background-color: #ffffff !important;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #eeeeee;
    }
    
    /* 5. ปุ่มวิเคราะห์สีน้ำเงิน */
    .stButton>button {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 10px;
        width: 100%;
        border: none;
    }

    /* 6. กล่อง Input Container */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1A1C24 !important;
    }

    /* 7. บังคับตัวหนังสือในส่วนคำแนะนำให้ชัดเจน */
    .analysis-text h5, .analysis-text p, .analysis-text li {
        color: #000000 !important;
    }
    div[data-testid="stNotification"] p {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ส่วนของ AI Logic (Fuzzy Logic) ---
temp_range = np.arange(34, 43.1, 0.1)
sleep_range = np.arange(0, 13, 1)
stress_range = np.arange(0, 11, 1)
risk_range = np.arange(0, 101, 1)

temp = ctrl.Antecedent(temp_range, 'temperature')
sleep = ctrl.Antecedent(sleep_range, 'sleep_hours')
stress = ctrl.Antecedent(stress_range, 'stress_level')
risk = ctrl.Consequent(risk_range, 'risk')

temp['normal'] = fuzz.trimf(temp.universe, [34, 36.5, 37.8])
temp['fever'] = fuzz.trapmf(temp.universe, [37.2, 38.5, 41, 43])
sleep['low'] = fuzz.trimf(sleep.universe, [0, 0, 6])
sleep['normal'] = fuzz.trimf(sleep.universe, [5, 8, 12])
stress['low'] = fuzz.trimf(stress.universe, [0, 0, 5])
stress['high'] = fuzz.trimf(stress.universe, [4, 7, 10])

risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
risk['medium'] = fuzz.trimf(risk.universe, [40, 60, 80])
risk['high'] = fuzz.trimf(risk.universe, [70, 85, 100])

rules = [
    ctrl.Rule(temp['fever'] & stress['high'], risk['high']),
    ctrl.Rule(temp['normal'] & sleep['normal'] & stress['low'], risk['low']),
    ctrl.Rule(temp['fever'] & sleep['low'], risk['high']),
    ctrl.Rule(temp['normal'] & (sleep['low'] | stress['high']), risk['medium']),
]
health_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

# --- 3. ส่วนหน้าจอหลัก ---
st.title("🏥 HealthRisk AI Advisor")
st.markdown("### ระบบประเมินความเสี่ยงสุขภาพด้วย *Mamdani Fuzzy Inference System*")
st.divider()

col_input, col_display = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("📋 ระบุข้อมูลตัวบ่งชี้")
    with st.container(border=True):
        in_temp = st.slider("🌡️ อุณหภูมิร่างกาย (°C)", 35.0, 42.0, 37.0, step=0.1)
        in_sleep = st.select_slider("😴 ชั่วโมงการนอนต่อวัน", options=range(13), value=7)
        in_stress = st.select_slider("🤯 ระดับความเครียด (0-10)", options=range(11), value=3)
    btn_calc = st.button("🚀 วิเคราะห์ผลลัพธ์")

with col_display:
    if btn_calc:
        health_sim.input['temperature'] = in_temp
        health_sim.input['sleep_hours'] = in_sleep
        health_sim.input['stress_level'] = in_stress
        
        try:
            health_sim.compute()
            res_risk = health_sim.output['risk']
            
            st.subheader("📊 ผลการวิเคราะห์")
            m1, m2 = st.columns(2)
            status = "ปลอดภัย" if res_risk < 40 else ("ควรระวัง" if res_risk < 70 else "อันตราย")
            m1.metric("ความเสี่ยงโดยรวม", f"{res_risk:.2f}%")
            m2.metric("สถานะสุขภาพ", status)

            st.markdown("<div class='analysis-text'>", unsafe_allow_html=True)
            st.write("##### **คำแนะนำจากระบบ:**") 
            
            if res_risk > 70:
                st.error("🚨 **สถานะ: อันตราย**")
                st.markdown("* ดื่มน้ำมากๆ และทานยาลดไข้\n* ควรพักผ่อนให้เพียงพอ\n* **โปรดพบแพทย์ทันที**")
            elif res_risk > 40:
                st.warning("⚠️ **สถานะ: ควรระวัง**")
                st.markdown("* ควรลดภาระงาน\n* พยายามนอนหลับให้ครบ 7-8 ชั่วโมง")
            else:
                st.success("✅ **สถานะ: ปลอดภัย**")
                st.markdown("* รักษาสุขภาพที่ดีต่อไป\n* ออกกำลังกายอย่างสม่ำเสมอ")
            
            st.markdown("</div>", unsafe_allow_html=True)

            # --- กราฟสรุปผล ---
            st.write("#### 📉 กราฟสรุปผล (Inference Visualization)")
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            ax.tick_params(colors='black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')

            ax.plot(risk_range, risk['low'].mf, 'g', linewidth=2, label='Low Risk')
            ax.plot(risk_range, risk['medium'].mf, 'orange', linewidth=2, label='Medium Risk')
            ax.plot(risk_range, risk['high'].mf, 'r', linewidth=2, label='High Risk')
            
            ax.axvline(x=res_risk, color='blue', linestyle='--', linewidth=2.5, label=f'Result ({res_risk:.1f}%)')
            
            ax.set_title("Health Risk Level", color='black', fontsize=14)
            ax.set_xlabel("Risk Percentage (%)", color='black')
            ax.set_ylabel("Membership Degree", color='black')
            ax.legend(loc='upper right', facecolor='white', edgecolor='black')
            ax.grid(True, linestyle=':', alpha=0.6)
            
            st.pyplot(fig)

        except Exception as e:
            st.error("ไม่สามารถคำนวณได้")
    else:
        st.info("กรุณาป้อนข้อมูลเพื่อเริ่มการวิเคราะห์")

# --- 4. ส่วนวิเคราะห์เชิงลึก ---
st.divider()
exp = st.expander("🛠️ ดูเบื้องหลังการทำงาน (Membership Degree Analysis)")
with exp:
    st.info("ค่าความเป็นสมาชิกตามหลักการ Fuzzy Logic")
    c1, c2, c3 = st.columns(3)
    deg_fever = fuzz.interp_membership(temp_range, temp['fever'].mf, in_temp)
    deg_sleep_low = fuzz.interp_membership(sleep_range, sleep['low'].mf, in_sleep)
    deg_stress_high = fuzz.interp_membership(stress_range, stress['high'].mf, in_stress)
    
    c1.markdown(f"ความเป็นสมาชิก 'ไข้': **{deg_fever:.2f}**")
    c2.markdown(f"ความเป็นสมาชิก 'นอนน้อย': **{deg_sleep_low:.2f}**")
    c3.markdown(f"ความเป็นสมาชิก 'เครียดสูง': **{deg_stress_high:.2f}**")
