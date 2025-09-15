Feature: Data Module System
  As a Vision AI developer
  I want flexible data loading and processing
  So that I can easily work with different datasets

  Background:
    Given the data module system is available
    And PyTorch Lightning data modules are supported

  Scenario: Base Data Module Creation
    Given I have the data module architecture
    When I create a new data module class
    Then it should inherit from the base data module
    And it should support standard dataset operations
    And it should integrate with PyTorch Lightning

  Scenario: CIFAR Data Module Setup
    Given the CIFAR data module is available
    When I initialize the CIFAR data module
    Then it should download the dataset if needed
    And it should apply proper transformations
    And it should create train, validation, and test splits

  Scenario: Data Transformations Pipeline
    Given a data module with transformations
    When I apply transformations to sample data
    Then the data should be properly preprocessed
    And the output should have correct dimensions
    And augmentations should be applied in training mode

  Scenario: Data Loading Performance
    Given a configured data module
    When I create data loaders
    Then they should load data efficiently
    And they should use appropriate batch sizes
    And they should utilize multiple workers when available

  Scenario: Custom Dataset Integration
    Given I have a custom dataset
    When I create a data module for it
    Then it should properly wrap the dataset
    And it should handle data loading correctly
    And it should support custom preprocessing

  Scenario: Data Module Validation
    Given a data module configuration
    When I validate the setup
    Then it should check data availability
    And it should validate transformation chains
    And it should verify batch dimensions

  Scenario: Data Split Configuration
    Given a data module with custom splits
    When I configure train/validation/test splits
    Then the splits should be properly created
    And they should maintain data integrity
    And they should be reproducible with seeds