# Model Update System

## Overview

The Quant News Trader implements an automated forward-rolling model improvement system that continuously updates and enhances trading models with new incoming data. This ensures models stay current and improve over time.

## Update Frequencies

### Micro Updates (Every 3 Days)
- **Scope**: Incremental learning on recent data
- **Data**: 30 days of new data + existing model
- **Method**: Weighted combination of old model (80%) + new data (20%)
- **Purpose**: Keep model current with latest market conditions

### Macro Updates (Every 7 Days)
- **Scope**: Full model retraining
- **Data**: Extended historical window (365 days)
- **Method**: Complete retrain from scratch
- **Purpose**: Incorporate broader market patterns and trends

### Major Updates (Every 30 Days)
- **Scope**: Hyperparameter optimization
- **Data**: Maximum historical data (540 days)
- **Method**: Randomized search CV for optimal parameters
- **Purpose**: Fine-tune model performance and robustness

### Epoch Updates (Every 365 Days)
- **Scope**: Complete model overhaul
- **Data**: Comprehensive historical dataset
- **Method**: Architecture comparison and selection
- **Purpose**: Consider fundamental model changes and improvements

## Configuration

Update settings are configured in `configs/config.yaml`:

```yaml
update:
  enabled: true
  base_lookback_days: 180

  # Update frequencies (in days)
  micro_update_days: 3
  macro_update_days: 7
  major_update_days: 30
  epoch_update_days: 365

  # Micro update settings
  micro_update:
    incremental: true
    new_data_days: 30
    retain_old_weight: 0.8

  # Macro update settings
  macro_update:
    full_retrain: true
    extended_lookback: 365
    feature_refresh: true

  # Major update settings
  major_update:
    hyperparameter_tuning: true
    architecture_search: false
    cross_validation_folds: 5

  # Epoch update settings
  epoch_update:
    complete_overhaul: true
    benchmark_competition: true
    feature_engineering: true
```

## Manual Updates

Run specific update types manually:

```bash
# Micro update
python -m src.model_update --update-type micro

# Macro update
python -m src.model_update --update-type macro

# Major update
python -m src.model_update --update-type major

# Epoch update
python -m src.model_update --update-type epoch
```

## Automated Scheduling

### Windows Task Scheduler

1. Open Task Scheduler
2. Create new task
3. Set trigger (daily at specific time)
4. Set action to run: `scripts\run_model_updates.bat`
5. Set working directory to project root

### PowerShell Script

The scheduler can also be run directly:

```powershell
.\scripts\schedule_model_updates.ps1
```

With options:
```powershell
# Force all updates
.\scripts\schedule_model_updates.ps1 -Force

# Force specific update type
.\scripts\schedule_model_updates.ps1 -UpdateType macro
```

## File Organization

All update-related files are properly organized:

```
models/
├── current_model.pkl          # Symlink to latest model
├── model_micro_20241201_143000.pkl
├── model_macro_20241201_120000.pkl
└── ...

reports/
├── model_update_micro_20241201_143000.json
├── model_update_macro_20241201_120000.json
└── ...

logs/
├── model_update_history.jsonl    # Update history log
└── model_update_scheduler_20241201.log
```

## Monitoring Updates

### Update History
Check `logs/model_update_history.jsonl` for complete update history:

```json
{"timestamp": "2024-12-01T14:30:00", "update_type": "micro", "performance": {"accuracy": 0.78}}
{"timestamp": "2024-12-01T12:00:00", "update_type": "macro", "performance": {"accuracy": 0.82}}
```

### Performance Tracking
Each update generates a detailed report in `reports/` with:
- Performance metrics before/after update
- Configuration used
- Model metadata
- Benchmarking results (for epoch updates)

### Scheduler Logs
Daily scheduler logs in `logs/model_update_scheduler_YYYYMMDD.log` show:
- Which updates were checked
- Which were due to run
- Execution results
- Any errors encountered

## Best Practices

1. **Monitor Performance**: Regularly review update reports to ensure models are improving
2. **Backup Models**: Important models are automatically saved with timestamps
3. **Resource Management**: Updates run sequentially to avoid resource conflicts
4. **Error Handling**: Failed updates don't break the system; next scheduled run will retry
5. **Data Quality**: Updates only run if sufficient quality data is available

## Troubleshooting

### Updates Not Running
- Check if `update.enabled: true` in config
- Verify scheduler is properly configured in Task Scheduler
- Check logs for error messages

### Poor Performance After Update
- Review update reports for performance metrics
- Check data quality and quantity
- Consider adjusting update parameters in config

### Resource Issues
- Updates may be resource-intensive, especially major/epoch updates
- Schedule during off-peak hours
- Monitor system resources during updates