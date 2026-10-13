# Actual test results

**No validated classifier exists yet. Classification results cannot be presented as performance.**

Generated from actual pytest JUnit records. Tests using synthetic, legacy and development-trained estimators exercise code only; they do not validate physiological or classification claims. Camera, printing and cross-browser behavior: **needs manual verification on device**.

| Python | Recorded run start | Pass | Fail | Skip | Pytest exit | Warnings | Frontend | pip check |
|---|---|---:|---:|---:|---:|---:|---|---|
| 3.13.0 | 2026-10-13T13:07:43.759919+05:30 | 146 | 0 | 0 | 0 | 11 | PASS | PASS |
| 3.14.5 | 2026-10-13T13:09:08.223301+05:30 | 144 | 0 | 2 | 0 | 11 | PASS | PASS |

## Environment and raw evidence

See `environment.md` for exact versions and recreation instructions. Plugin autoload was disabled, bytecode writes disabled and default preview flag false for the suite; individual tests explicitly enable preview. The application does not suppress warnings. Warning counts are read from the actual logs linked below; inspect those logs for warning categories. The per-test table includes NumPy/joblib deserialization checks.

- Python 3.13.0; macOS-26.5.1-arm64-arm-64bit-Mach-O; Node v24.15.0. [Environment record](../reports/test_run_py3.13.0_2026-10-13_130743_ff61c905.json), [JUnit XML](../reports/test_run_py3.13.0_2026-10-13_130743_ff61c905.xml), [test log](../reports/test_run_py3.13.0_2026-10-13_130743_ff61c905.log).
- Python 3.14.5; macOS-26.5.1-arm64-arm-64bit-Mach-O; Node v24.15.0. [Environment record](../reports/test_run_py3.14.5_2026-10-13_130908_d98ce7fb.json), [JUnit XML](../reports/test_run_py3.14.5_2026-10-13_130908_d98ce7fb.xml), [test log](../reports/test_run_py3.14.5_2026-10-13_130908_d98ce7fb.log).

## Per-test outcomes

| Test | 3.13.0 | 3.14.5 |
|---|---|---|
| `tests.test_stage1::test_bad_file` | PASS | PASS |
| `tests.test_stage1::test_cli_startup_error` | PASS | PASS |
| `tests.test_stage1::test_cwd_independent_model_loading` | PASS | PASS |
| `tests.test_stage1::test_disallowed_extension` | PASS | PASS |
| `tests.test_stage1::test_disallowed_format` | PASS | PASS |
| `tests.test_stage1::test_empty_filename` | PASS | PASS |
| `tests.test_stage1::test_empty_overlay_regions[color0]` | PASS | PASS |
| `tests.test_stage1::test_empty_overlay_regions[color1]` | PASS | PASS |
| `tests.test_stage1::test_empty_overlay_regions[color2]` | PASS | PASS |
| `tests.test_stage1::test_healthy_prediction_regression[healthy_006.JPG]` | PASS | PASS |
| `tests.test_stage1::test_healthy_prediction_regression[healthy_008.JPG]` | PASS | PASS |
| `tests.test_stage1::test_healthy_visualization_regression[healthy_006.JPG]` | PASS | PASS |
| `tests.test_stage1::test_healthy_visualization_regression[healthy_008.JPG]` | PASS | PASS |
| `tests.test_stage1::test_http_errors_json` | PASS | PASS |
| `tests.test_stage1::test_insufficient_dataset_does_not_write_artifacts` | PASS | PASS |
| `tests.test_stage1::test_invalid_upload_cleanup` | PASS | PASS |
| `tests.test_stage1::test_model_failure` | PASS | PASS |
| `tests.test_stage1::test_model_load_failure[False]` | PASS | PASS |
| `tests.test_stage1::test_model_load_failure[True]` | PASS | PASS |
| `tests.test_stage1::test_no_file` | PASS | PASS |
| `tests.test_stage1::test_no_leaf` | PASS | PASS |
| `tests.test_stage1::test_originals_and_backups_unchanged` | PASS | PASS |
| `tests.test_stage1::test_pixel_limit_before_decode` | PASS | PASS |
| `tests.test_stage1::test_probability_schema_failure` | PASS | PASS |
| `tests.test_stage1::test_safe_server_defaults` | PASS | PASS |
| `tests.test_stage1::test_schema_failure` | PASS | PASS |
| `tests.test_stage1::test_unexpected_exception` | PASS | PASS |
| `tests.test_stage1::test_unique_safe_path_and_cleanup[False]` | PASS | PASS |
| `tests.test_stage1::test_unique_safe_path_and_cleanup[True]` | PASS | PASS |
| `tests.test_stage1::test_upload_byte_limit` | PASS | PASS |
| `tests.test_stage1::test_versioned_paths` | PASS | PASS |
| `tests.test_stage2::test_comparisons_are_pooled_descriptions_not_importance` | PASS | PASS |
| `tests.test_stage2::test_guidance_asks_for_independent_checks[Disease_Confounded]` | PASS | PASS |
| `tests.test_stage2::test_guidance_asks_for_independent_checks[Healthy]` | PASS | PASS |
| `tests.test_stage2::test_guidance_asks_for_independent_checks[Moderate_Water_Stress]` | PASS | PASS |
| `tests.test_stage2::test_guidance_asks_for_independent_checks[Severe_Water_Stress]` | PASS | PASS |
| `tests.test_stage2::test_guidance_asks_for_independent_checks[unknown]` | PASS | PASS |
| `tests.test_stage2::test_indicator_integer_and_confounded_suppression` | PASS | PASS |
| `tests.test_stage2::test_prediction_contract[Disease_Confounded]` | PASS | PASS |
| `tests.test_stage2::test_prediction_contract[Healthy]` | PASS | PASS |
| `tests.test_stage2::test_shared_standardized_features_and_rejections` | PASS | PASS |
| `tests.test_stage2::test_visible_copy_and_srs_suggestions` | PASS | PASS |
| `tests.test_stage3::test_abstain_before_probability_rounding[0.5996-uncertain]` | PASS | PASS |
| `tests.test_stage3::test_abstain_before_probability_rounding[0.6-classified]` | PASS | PASS |
| `tests.test_stage3::test_abstain_before_probability_rounding[0.8-classified]` | PASS | PASS |
| `tests.test_stage3::test_clipped_contour` | PASS | PASS |
| `tests.test_stage3::test_components_before_largest_selection` | PASS | PASS |
| `tests.test_stage3::test_configurable_uncertainty_threshold` | PASS | PASS |
| `tests.test_stage3::test_dataset_report_and_overlapping_counts` | PASS | PASS |
| `tests.test_stage3::test_full_frame_multiple_reasons` | PASS | PASS |
| `tests.test_stage3::test_gate_is_not_semantic_leaf_detection` | PASS | PASS |
| `tests.test_stage3::test_legacy_preprocessing_refused` | PASS | PASS |
| `tests.test_stage3::test_mask_empty` | PASS | PASS |
| `tests.test_stage3::test_rejection_skips_model_and_provides_preview[color0]` | PASS | PASS |
| `tests.test_stage3::test_rejection_skips_model_and_provides_preview[color1]` | PASS | PASS |
| `tests.test_stage3::test_rejection_skips_model_and_provides_preview[color2]` | PASS | PASS |
| `tests.test_stage3::test_scaled_kernel` | PASS | PASS |
| `tests.test_stage3::test_small_and_tiny` | PASS | PASS |
| `tests.test_stage3::test_small_speck_is_not_second_leaf` | PASS | PASS |
| `tests.test_stage3::test_threshold_boundaries` | PASS | PASS |
| `tests.test_stage3::test_unreadable_pipeline` | PASS | PASS |
| `tests.test_stage3::test_working_resolution[size0]` | PASS | PASS |
| `tests.test_stage3::test_working_resolution[size1]` | PASS | PASS |
| `tests.test_stage3::test_working_resolution[size2]` | PASS | PASS |
| `tests.test_stage3::test_working_resolution[size3]` | PASS | PASS |
| `tests.test_stage4::test_blocked_reports_and_legacy_filenames_separate` | PASS | PASS |
| `tests.test_stage4::test_border_perimeter_fraction` | PASS | PASS |
| `tests.test_stage4::test_builder_fails_cleanly_without_partial_csv` | PASS | PASS |
| `tests.test_stage4::test_builder_reads_identity_fields` | PASS | PASS |
| `tests.test_stage4::test_candidates_scaling_and_rf_regularization` | PASS | PASS |
| `tests.test_stage4::test_cross_label_duplicates_fail` | PASS | PASS |
| `tests.test_stage4::test_current_dataset_stops_before_folds_or_fits` | PASS | PASS |
| `tests.test_stage4::test_duplicate_identity_conflict_and_blanks` | PASS | PASS |
| `tests.test_stage4::test_error_metrics_definition` | PASS | PASS |
| `tests.test_stage4::test_explicit_duplicate_leakage_detection` | PASS | PASS |
| `tests.test_stage4::test_fallback_session_links_multiple_plants_conservatively` | PASS | PASS |
| `tests.test_stage4::test_full_precision_tie_break` | PASS | PASS |
| `tests.test_stage4::test_group_and_duplicate_separation_in_every_nested_fold` | PASS | PASS |
| `tests.test_stage4::test_import_builder_has_no_writes` | PASS | PASS |
| `tests.test_stage4::test_legacy_numbers_reproduce_protected_original` | PASS | PASS |
| `tests.test_stage4::test_manifest_counts_human_blanks_and_stage3_values` | PASS | PASS |
| `tests.test_stage4::test_manifest_gate_status_cannot_bypass_gate` | PASS | PASS |
| `tests.test_stage4::test_manifest_generation_does_not_overwrite` | PASS | PASS |
| `tests.test_stage4::test_missing_group_blocks_no_image_fallback` | PASS | PASS |
| `tests.test_stage4::test_nested_success_runs_selection_and_is_reproducible` | PASS | PASS |
| `tests.test_stage4::test_plant_session_fallback_and_namespace` | PASS | PASS |
| `tests.test_stage4::test_successful_builder_uses_groups_and_sidecar` | PASS | PASS |
| `tests.test_stage5::test_ablation_nested_groups_and_identical_folds` | PASS | PASS |
| `tests.test_stage5::test_api_includes_faithful_explanation` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[classes_-value9]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[cv_protocol-value8]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[dataset_hash-unknown]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[extractor_version-wrong]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[feature_names_in_-value4]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[feature_schema-value3]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[library_versions-value6]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[mask_quality-value10]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[preprocessing_version-wrong]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[schema_version-wrong]` | PASS | PASS |
| `tests.test_stage5::test_contract_rejects_mismatch[training_date-]` | PASS | PASS |
| `tests.test_stage5::test_exact_logistic_explanation` | PASS | PASS |
| `tests.test_stage5::test_experimental_feature_math` | PASS | PASS |
| `tests.test_stage5::test_new_bundle_round_trip` | PASS | PASS |
| `tests.test_stage5::test_original_load_has_no_numpy_joblib_warnings` | PASS | PASS |
| `tests.test_stage5::test_original_refused_at_load` | PASS | PASS |
| `tests.test_stage5::test_real_ablation_blocked_before_any_fit` | PASS | PASS |
| `tests.test_stage5::test_training_writer_persists_contract_without_touching_production` | PASS | PASS |
| `tests.test_stage6::test_estimator_and_schema_mismatches[comparison_class_order]` | PASS | PASS |
| `tests.test_stage6::test_estimator_and_schema_mismatches[estimator_order]` | PASS | PASS |
| `tests.test_stage6::test_estimator_and_schema_mismatches[extra_feature]` | PASS | PASS |
| `tests.test_stage6::test_preview_rejections_and_repeatability` | PASS | PASS |
| `tests.test_stage6::test_preview_skips_even_available_model_and_hides_benchmarks` | PASS | PASS |
| `tests.test_stage6::test_real_image_features_repeat_exactly` | PASS | PASS |
| `tests.test_stage6::test_rgb_order_gli_bounds_exg_identity_and_empty_mask` | PASS | PASS |
| `tests.test_stage6::test_segmentation_acquisition_failures[blank]` | PASS | PASS |
| `tests.test_stage6::test_segmentation_acquisition_failures[bright]` | PASS | PASS |
| `tests.test_stage6::test_segmentation_acquisition_failures[clipped]` | PASS | PASS |
| `tests.test_stage6::test_segmentation_acquisition_failures[complex]` | PASS | PASS |
| `tests.test_stage6::test_segmentation_acquisition_failures[low_light]` | PASS | PASS |
| `tests.test_stage6::test_segmentation_acquisition_failures[multiple]` | PASS | PASS |
| `tests.test_stage6::test_shape_rectangle_circle_and_zero_perimeter` | PASS | PASS |
| `tests.test_stage6::test_startup_without_model_in_subprocess[false-False]` | PASS | PASS |
| `tests.test_stage6::test_startup_without_model_in_subprocess[true-True]` | PASS | PASS |
| `tests.test_stage6::test_timing_rejects_invalid_repeats` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[1-color0]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[1-color1]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[1-color2]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[10-color0]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[10-color1]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[10-color2]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[2-color0]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[2-color1]` | PASS | PASS |
| `tests.test_stage6::test_tiny_color_visualizations_no_empty_overlay_crash[2-color2]` | PASS | PASS |
| `tests.test_stage6::test_truncated_image_json` | PASS | PASS |
| `tests.test_wholeplant::test_empty_mask_and_decode_errors` | PASS | PASS |
| `tests.test_wholeplant::test_extractor_reproducibility_and_empty_bands` | PASS | PASS |
| `tests.test_wholeplant::test_identity_and_warm_background_limit` | PASS | PASS |
| `tests.test_wholeplant::test_manifest_no_invented_identity_and_conflicts` | PASS | PASS |
| `tests.test_wholeplant::test_multiple_foliage_regions_are_retained` | PASS | PASS |
| `tests.test_wholeplant::test_reject_blank_accept_yellow_flag_clipped[blank]` | PASS | PASS |
| `tests.test_wholeplant::test_reject_blank_accept_yellow_flag_clipped[clipped]` | PASS | PASS |
| `tests.test_wholeplant::test_reject_blank_accept_yellow_flag_clipped[yellow]` | PASS | PASS |
| `tests.test_wholeplant::test_saved_model_contract_and_coefficients` | PASS | SKIP |
| `tests.test_wholeplant::test_tiny_uniform_green_rejected` | PASS | PASS |
| `tests.test_wholeplant::test_training_requires_explicit_development_mode_and_samples` | PASS | PASS |
| `tests.test_wholeplant::test_wholeplant_flask_all_images_and_preview` | PASS | SKIP |

## Interpretation

No executed automated failures in a PASS-only run does not imply scientific readiness. Original single-leaf grouped evaluation remains blocked by missing accepted classes and group provenance. The new whole-plant dataset has all four classes but still lacks verified group/label provenance; its trained classifier is experimental, not validated. The original classifier remains incompatible. Python 3.14 skips the two saved whole-plant deployment tests because that artifact requires its exact Python 3.13 environment. Frontend checks use a minimal DOM, not real browsers. Whole-plant functional timing is documented separately in wholeplant_implementation.md; neither timing nor training-image inference establishes predictive validity.
