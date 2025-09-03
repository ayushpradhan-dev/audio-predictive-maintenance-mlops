# Configure Terraform provider for Azure
terraform {
    required_providers {
        azurerm = {
            source = "hashicorp/azurerm"
            version = "~>3.0"
        }
    }
}

# Configure the Azure provider to use credentials from the Azure CLI
provider "azurerm" {
    features{}
}

# Define variables
variable "resource_group_name" {
    type = string
    description = "The name of the Azure Resource Group."
    default = "rg-audio-mlops"
}

variable "location" {
    type = string
    description = "The Azure region where resources will be created."
    default = "UK South"
}

variable "acr_name" {
    type = string
    description = "A globally unique name for the Azure Container Registry."
    default = "acraudiomlops"
}

variable "aci_name" {
    type = string
    description = "A globally unique name for the DNS of the Azure Container Instance."
    default = "aci-audio-mlops"
}

# Create Resource Group, logical container for Azure resources.
resource "azurerm_resource_group" "rg" {
    name = var.resource_group_name
    location = var.location
}

# Create Azure Container Registry (ACR)
# Private Docker registry to store container image.
resource "azurerm_container_registry" "acr" {
    name = var.acr_name
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location

    sku = "Basic"
    admin_enabled = true
}

# Create Azure Container Instance (ACI)
# Pulls image from ACR and run it as a public web service.
resource "azurerm_container_group" "aci" {
    name = "aci-audio-mlops"
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location

    # The image to pull. This uses the login server from the ACR.connection {
    # Image on Docker Hub is named 'ayushpradhan24/audio-predictive-maintenance'
    image_registry_credential {
        server = azurerm_container_registry.acr.login_server
        username = azurerm_container_registry.acr.admin_username
        password = azurerm_container_registry.acr.admin_password
    }
    
    container {
        name = "api-container"
        # For now pulls directly from public docker hub repo, later will pull from ACR.
        image = "ayushpradhan24/audio-predictive-maintenance"

        cpu = 1
        memory = 1.5

        ports {
            port = 8000
            protocol = "TCP"
        }
    }

    os_type = "Linux"
    ip_address_type = "Public"
    dns_name_label = var.aci_name
}

# Output API URL
# Print public URL of the running API
output "api_url" {
    value = "http://${azurerm_container_group.aci.fqdn}:8000/docs"
}