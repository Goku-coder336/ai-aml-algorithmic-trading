
ax3 = axes[2]
importance    = rf_model.feature_importances_
sorted_idx    = np.argsort(importance)
colors        = [ACCENT if importance[i] > 0.1 else BLUE for i in sorted_idx]
ax3.barh([display_names[i] for i in sorted_idx], importance[sorted_idx], color=colors)
ax3.set_title('XAI Feature Importance\nfor AML Detection', color='white')
ax3.set_xlabel('Importance Score')
ax3.grid(True, axis='x')

plt.tight_layout()
plt.savefig('03_aml_detection.png', dpi=150, bbox_inches='tight')
plt.show()

def explain_transaction(transaction_idx, model, scaler, X, feature_names):
    transaction = X.iloc[[transaction_idx]]
    scaled      = scaler.transform(transaction)
    probability = model.predict_proba(scaled)[0][1]
    base_prob   = probability
    contributions = []
    for i in range(len(feature_names)):
        modified       = scaled.copy()
        modified[0, i] = 0
        new_prob       = model.predict_proba(modified)[0][1]
        contributions.append(base_prob - new_prob)
    return probability, contributions

feature_names = ['transaction_amount', 'trade_frequency', 'time_of_day', 'counterparties',
                 'cross_border', 'round_amount', 'rapid_succession', 'pre_announcement']

X_df    = transactions[feature_names].reset_index(drop=True)
sus_idx = transactions[transactions.label == 1].index[0]
prob, contributions = explain_transaction(sus_idx, rf_model, scaler, X_df, feature_names)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
ax1.axis('off')
transaction_row = transactions.iloc[sus_idx]
table_data = [[name, f'{transaction_row[feat]:.2f}']
              for feat, name in zip(feature_names, display_names)]
table = ax1.table(cellText=table_data, colLabels=['Feature', 'Value'],
                  cellLoc='left', loc='center', colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor('#1a1a2e' if row % 2 == 0 else '#0d1117')
    cell.set_text_props(color='white')
    cell.set_edgecolor('#333')
    if row == 0:
        cell.set_facecolor('#00cc7a')
        cell.set_text_props(color='black', fontweight='bold')
ax1.set_title(f'Transaction Profile\nSuspicion Score: {prob:.1%}', color='white', fontsize=13, pad=20)

ax2 = axes[1]
sorted_contrib  = sorted(zip(contributions, display_names), reverse=True)
vals, names     = zip(*sorted_contrib)
bar_colors      = [RED if v > 0 else BLUE for v in vals]
ax2.barh(names, vals, color=bar_colors, alpha=0.85)
ax2.axvline(0, color='white', lw=0.8)
ax2.set_title('XAI — Why Was This Transaction Flagged?', color='white', fontsize=12)
ax2.set_xlabel('Feature Contribution to Suspicion Score')
ax2.grid(True, axis='x', alpha=0.3)
red_patch  = mpatches.Patch(color=RED,  label='Increases suspicion')
blue_patch = mpatches.Patch(color=BLUE, label='Decreases suspicion')
ax2.legend(handles=[red_patch, blue_patch])

plt.tight_layout()
plt.savefig('04_xai_explanation.png', dpi=150, bbox_inches='tight')
plt.show()
n_minutes   = 60
price       = np.zeros(n_minutes)
price[0]    = 100.0
time_steps  = np.arange(n_minutes)
crash_start = 45
volume_normal, volume_rogue = [], []

for t in range(1, n_minutes):
    normal_impact = np.random.normal(0, 0.05)
    if t >= crash_start:
        rogue_orders  = np.random.choice([-1, 1]) * np.random.exponential(2)
        rogue_volume  = np.random.randint(500, 2000)
        normal_volume = np.random.randint(10, 100)
        cascade       = -abs(rogue_orders) * 0.8
        price[t]      = price[t-1] * (1 + normal_impact * 0.001 + cascade * 0.01)
    else:
        rogue_volume  = np.random.randint(10, 80)
        normal_volume = np.random.randint(20, 120)
        price[t]      = price[t-1] * (1 + normal_impact * 0.001)
    volume_rogue.append(rogue_volume)
    volume_normal.append(normal_volume)

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

ax1 = axes[0]
ax1.plot(time_steps, price, color=ACCENT, lw=2)
ax1.axvline(crash_start, color=RED, ls='--', lw=2, label=f'Rogue Algo Triggered (min {crash_start})')
ax1.fill_between(time_steps[crash_start:], price[crash_start:], price[crash_start],
                 alpha=0.2, color=RED)
ax1.set_title('Flash Crash Simulation — Knight Capital Style', color='white', fontsize=12)
ax1.set_ylabel('Price ($)')
ax1.legend()
ax1.grid(True)

ax2 = axes[1]
ax2.bar(time_steps[1:], volume_normal, label='Normal HFT Volume', color=BLUE, alpha=0.7)
ax2.bar(time_steps[1:], volume_rogue,  label='Rogue Algo Volume',  color=RED,  alpha=0.7,
        bottom=volume_normal)
ax2.axvline(crash_start, color=RED, ls='--', lw=2)
ax2.set_title('Trading Volume — Rogue Algorithm Dominates at Crash', color='white', fontsize=12)
ax2.set_xlabel('Minutes')
ax2.set_ylabel('Order Volume')
ax2.legend()
ax2.grid(True, axis='y')

plt.tight_layout()
plt.savefig('05_flash_crash.png', dpi=150, bbox_inches='tight')
plt.show()

price_drop = ((price[crash_start] - min(price[crash_start:])) / price[crash_start]) * 100
print(f'Price at crash start:  ${price[crash_start]:.2f}')
print(f'Lowest point:          ${min(price[crash_start:]):.2f}')
print(f'Maximum drawdown:      -{price_drop:.1f}% in {n_minutes - crash_start} minutes')
