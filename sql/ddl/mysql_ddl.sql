-- MySQL Script generated by MySQL Workbench
-- Tue 07 May 2019 05:10:27 PM CDT
-- Model: New Model    Version: 1.0
-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema lineage
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema lineage
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `lineage` ;
USE `lineage` ;

-- -----------------------------------------------------
-- Table `lineage`.`experiment`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`experiment` (
  `id` INT NOT NULL,
  `create_time` VARCHAR(45) NOT NULL,
  `start_time` DATETIME NOT NULL,
  `parameters` JSON NULL,
  `commit_hash` VARCHAR(2000) NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`workflow`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`workflow` (
  `id` INT NOT NULL,
  `directory_path` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`id`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`artifact`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`artifact` (
  `id` INT NOT NULL,
  `filename` VARCHAR(100) NOT NULL,
  `path` VARCHAR(200) NULL,
  `workflow_id` INT NOT NULL,
  PRIMARY KEY (`id`, `workflow_id`),
  INDEX `fk_Artifact_Workflow_idx` (`workflow_id` ASC) VISIBLE,
  CONSTRAINT `fk_artifact_workflow`
    FOREIGN KEY (`workflow_id`)
    REFERENCES `lineage`.`workflow` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`artifact_column`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`artifact_column` (
  `id` INT NOT NULL,
  `column_name` VARCHAR(45) NOT NULL,
  `column_type` VARCHAR(45) NULL,
  `sketch_signature` VARCHAR(45) NULL,
  `artifact_id` INT NOT NULL,
  `Time Taken` INT NULL,
  PRIMARY KEY (`id`, `artifact_id`),
  INDEX `fk_ArtifactColumn_Artifact1_idx` (`artifact_id` ASC) VISIBLE,
  CONSTRAINT `fk_artifact_column_artifact1`
    FOREIGN KEY (`artifact_id`)
    REFERENCES `lineage`.`artifact` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`ground_truth_edge`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`ground_truth_edge` (
  `id` INT NOT NULL,
  `artifact_1` INT NULL,
  `artifact_2` INT NULL,
  `workflow_id` INT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX `fk_GroundTruthEdge_Artifact1_idx` (`artifact_1` ASC) VISIBLE,
  INDEX `fk_GroundTruthEdge_Workflow1_idx` (`workflow_id` ASC) VISIBLE,
  CONSTRAINT `fk_ground_truth_edge_artifact1`
    FOREIGN KEY (`artifact_1`)
    REFERENCES `lineage`.`artifact` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_ground_truth_edge_artifact2`
    FOREIGN KEY (`artifact_1`)
    REFERENCES `lineage`.`artifact` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_ground_truth_edge_workflow`
    FOREIGN KEY (`workflow_id`)
    REFERENCES `lineage`.`workflow` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`cluster`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`cluster` (
  `id` INT NOT NULL,
  `artifact_id` INT NOT NULL,
  `type` VARCHAR(45) NULL,
  `experiment_id` INT NOT NULL,
  PRIMARY KEY (`id`, `artifact_id`),
  INDEX `fk_PreCluster_Artifact_Artifact1_idx` (`artifact_id` ASC) VISIBLE,
  INDEX `fk_Cluster_Experiment1_idx` (`experiment_id` ASC) VISIBLE,
  CONSTRAINT `fk_cluster_artifact`
    FOREIGN KEY (`artifact_id`)
    REFERENCES `lineage`.`artifact` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_cluster_experiment`
    FOREIGN KEY (`experiment_id`)
    REFERENCES `lineage`.`experiment` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`relationship_edge`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`relationship_edge` (
  `id` INT NOT NULL,
  `artifact_1` INT NOT NULL,
  `artifact_2` INT NOT NULL,
  `distance_type` VARCHAR(45) NULL,
  `distance_value` FLOAT NULL,
  `experiment_id` INT NOT NULL,
  PRIMARY KEY (`id`, `experiment_id`),
  INDEX `fk_PointPreservingEdge_Experiment1_idx` (`experiment_id` ASC) VISIBLE,
  INDEX `fk_InferedEdge_Artifact1_idx` (`artifact_1` ASC) VISIBLE,
  INDEX `fk_InferedEdge_Artifact2_idx` (`artifact_2` ASC) VISIBLE,
  CONSTRAINT `fk_experiment`
    FOREIGN KEY (`experiment_id`)
    REFERENCES `lineage`.`experiment` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_relationship_edge_artifact1`
    FOREIGN KEY (`artifact_1`)
    REFERENCES `lineage`.`artifact` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `fk_relationship_edge_artifact2`
    FOREIGN KEY (`artifact_2`)
    REFERENCES `lineage`.`artifact` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `lineage`.`time_log`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `lineage`.`time_log` (
  `task_name` VARCHAR(45) NULL,
  `time_taken` INT NULL,
  `experiment_id` INT NULL,
  INDEX `fk_TimeLog_Experiment1_idx` (`experiment_id` ASC) VISIBLE,
  CONSTRAINT `fk_TimeLog_Experiment1`
    FOREIGN KEY (`experiment_id`)
    REFERENCES `lineage`.`experiment` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
