var __defProp = Object.defineProperty;
var __defProps = Object.defineProperties;
var __getOwnPropDescs = Object.getOwnPropertyDescriptors;
var __getOwnPropSymbols = Object.getOwnPropertySymbols;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __propIsEnum = Object.prototype.propertyIsEnumerable;
var __knownSymbol = (name, symbol) => (symbol = Symbol[name]) ? symbol : Symbol.for("Symbol." + name);
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __spreadValues = (a, b) => {
  for (var prop in b || (b = {}))
    if (__hasOwnProp.call(b, prop))
      __defNormalProp(a, prop, b[prop]);
  if (__getOwnPropSymbols)
    for (var prop of __getOwnPropSymbols(b)) {
      if (__propIsEnum.call(b, prop))
        __defNormalProp(a, prop, b[prop]);
    }
  return a;
};
var __spreadProps = (a, b) => __defProps(a, __getOwnPropDescs(b));
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
var __async = (__this, __arguments, generator) => {
  return new Promise((resolve, reject) => {
    var fulfilled = (value) => {
      try {
        step(generator.next(value));
      } catch (e) {
        reject(e);
      }
    };
    var rejected = (value) => {
      try {
        step(generator.throw(value));
      } catch (e) {
        reject(e);
      }
    };
    var step = (x) => x.done ? resolve(x.value) : Promise.resolve(x.value).then(fulfilled, rejected);
    step((generator = generator.apply(__this, __arguments)).next());
  });
};
var __await = function(promise, isYieldStar) {
  this[0] = promise;
  this[1] = isYieldStar;
};
var __asyncGenerator = (__this, __arguments, generator) => {
  var resume = (k, v, yes, no) => {
    try {
      var x = generator[k](v), isAwait = (v = x.value) instanceof __await, done = x.done;
      Promise.resolve(isAwait ? v[0] : v).then((y) => isAwait ? resume(k === "return" ? k : "next", v[1] ? { done: y.done, value: y.value } : y, yes, no) : yes({ value: y, done })).catch((e) => resume("throw", e, yes, no));
    } catch (e) {
      no(e);
    }
  }, method = (k) => it[k] = (x) => new Promise((yes, no) => resume(k, x, yes, no)), it = {};
  return generator = generator.apply(__this, __arguments), it[__knownSymbol("asyncIterator")] = () => it, method("next"), method("throw"), method("return"), it;
};
var __forAwait = (obj, it, method) => (it = obj[__knownSymbol("asyncIterator")]) ? it.call(obj) : (obj = obj[__knownSymbol("iterator")](), it = {}, method = (key, fn) => (fn = obj[key]) && (it[key] = (arg) => new Promise((yes, no, done) => (arg = fn.call(obj, arg), done = arg.done, Promise.resolve(arg.value).then((value) => yes({ value, done }), no)))), method("next"), method("return"), it);

// src/glue/messages.ts
var GLUE_VERSION = 1;
var GLUE_MESSAGE_PROTOTYPES = {
  "erro_evt": {
    "name": "erro_evt",
    "structName": "glue_msg_error",
    "className": "GlueMsgError",
    "fields": [
      {
        "type": "str",
        "name": "message",
        "isNullable": false
      }
    ]
  },
  "load_req": {
    "name": "load_req",
    "structName": "glue_msg_load_req",
    "className": "GlueMsgLoadReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "model_paths",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "mmproj_path",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "n_ctx_auto",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mmap",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "use_mlock",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_gpu_layers",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_threads",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "model_alias",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "log_level",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "embeddings",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "offload_kqv",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_parallel",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "pooling_type",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "rope_scaling_type",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_base",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "rope_freq_scale",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_ext_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_attn_factor",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_fast",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "yarn_beta_slow",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "yarn_orig_ctx",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_k",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "cache_type_v",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "kv_unified",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "flash_attn",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "swa_full",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_ctx_checkpoints",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "checkpoint_min_step",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "chat_template",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "jinja",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "default_template_kwargs_vals",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "reasoning",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_min_tokens",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "image_max_tokens",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "warmup",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "no_kv_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "mmproj_offload",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cont_batching",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_keep",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "ctx_shift",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "cache_idle_slots",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "n_cache_reuse",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "lora_paths",
        "isNullable": true
      },
      {
        "type": "arr_float",
        "name": "lora_scales",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "lora_init_without_apply",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "native_state_paths",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "spec_draft_model",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_ngl",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_max",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_n_min",
        "isNullable": true
      },
      {
        "type": "float",
        "name": "spec_draft_p_min",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "spec_draft_threads_batch",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_keys",
        "isNullable": true
      },
      {
        "type": "arr_str",
        "name": "kv_overrides_vals",
        "isNullable": true
      },
      {
        "type": "int",
        "name": "reasoning_budget_tokens",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_budget_message",
        "isNullable": true
      },
      {
        "type": "str",
        "name": "reasoning_format",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "skip_chat_parsing",
        "isNullable": true
      },
      {
        "type": "bool",
        "name": "prefill_assistant",
        "isNullable": true
      }
    ]
  },
  "load_res": {
    "name": "load_res",
    "structName": "glue_msg_load_res",
    "className": "GlueMsgLoadRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_batch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ubatch",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_vocab",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_ctx_train",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_embd",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "n_layer",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_key",
        "isNullable": false
      },
      {
        "type": "arr_str",
        "name": "metadata_val",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_bos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eos",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_eot",
        "isNullable": false
      },
      {
        "type": "arr_int",
        "name": "list_tokens_eog",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_bos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "add_eos_token",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_encoder",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "token_decoder_start",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "media_marker",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_image_input",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_audio_input",
        "isNullable": false
      }
    ]
  },
  "cmpl_req": {
    "name": "cmpl_req",
    "structName": "glue_msg_completion_req",
    "className": "GlueMsgCompletionReq",
    "fields": [
      {
        "type": "bool",
        "name": "is_chat",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "cmpl_res": {
    "name": "cmpl_res",
    "structName": "glue_msg_completion_res",
    "className": "GlueMsgCompletionRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "embd_req": {
    "name": "embd_req",
    "structName": "glue_msg_embedding_req",
    "className": "GlueMsgEmbeddingReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      },
      {
        "type": "arr_raw",
        "name": "files",
        "isNullable": false
      }
    ]
  },
  "embd_res": {
    "name": "embd_res",
    "structName": "glue_msg_embedding_res",
    "className": "GlueMsgEmbeddingRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "rrnk_req": {
    "name": "rrnk_req",
    "structName": "glue_msg_rerank_req",
    "className": "GlueMsgRerankReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "rrnk_res": {
    "name": "rrnk_res",
    "structName": "glue_msg_rerank_res",
    "className": "GlueMsgRerankRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "sys1_req": {
    "name": "sys1_req",
    "structName": "glue_msg_systemone_req",
    "className": "GlueMsgSystemoneReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "sys1_res": {
    "name": "sys1_res",
    "structName": "glue_msg_systemone_res",
    "className": "GlueMsgSystemoneRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "scor_req": {
    "name": "scor_req",
    "structName": "glue_msg_score_req",
    "className": "GlueMsgScoreReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "scor_res": {
    "name": "scor_res",
    "structName": "glue_msg_score_res",
    "className": "GlueMsgScoreRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "slot_req": {
    "name": "slot_req",
    "structName": "glue_msg_slots_req",
    "className": "GlueMsgSlotsReq",
    "fields": [
      {
        "type": "int",
        "name": "id_slot",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "action",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "filename",
        "isNullable": false
      }
    ]
  },
  "slot_res": {
    "name": "slot_res",
    "structName": "glue_msg_slots_res",
    "className": "GlueMsgSlotsRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "tmpl_req": {
    "name": "tmpl_req",
    "structName": "glue_msg_apply_template_req",
    "className": "GlueMsgApplyTemplateReq",
    "fields": [
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "tmpl_res": {
    "name": "tmpl_res",
    "structName": "glue_msg_apply_template_res",
    "className": "GlueMsgApplyTemplateRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "prompt",
        "isNullable": false
      }
    ]
  },
  "gres_req": {
    "name": "gres_req",
    "structName": "glue_msg_get_result_req",
    "className": "GlueMsgGetResultReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "gres_res": {
    "name": "gres_res",
    "structName": "glue_msg_get_result_res",
    "className": "GlueMsgGetResultRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "has_more",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "is_error",
        "isNullable": false
      },
      {
        "type": "str",
        "name": "data_json",
        "isNullable": false
      }
    ]
  },
  "cncl_req": {
    "name": "cncl_req",
    "structName": "glue_msg_cancel_req",
    "className": "GlueMsgCancelReq",
    "fields": [
      {
        "type": "int",
        "name": "req_id",
        "isNullable": false
      }
    ]
  },
  "cncl_res": {
    "name": "cncl_res",
    "structName": "glue_msg_cancel_res",
    "className": "GlueMsgCancelRes",
    "fields": [
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  },
  "tbop_req": {
    "name": "tbop_req",
    "structName": "glue_msg_test_backend_ops_req",
    "className": "GlueMsgTestBackendOpsReq",
    "fields": [
      {
        "type": "arr_str",
        "name": "args",
        "isNullable": false
      }
    ]
  },
  "tbop_res": {
    "name": "tbop_res",
    "structName": "glue_msg_test_backend_ops_res",
    "className": "GlueMsgTestBackendOpsRes",
    "fields": [
      {
        "type": "int",
        "name": "retcode",
        "isNullable": false
      },
      {
        "type": "bool",
        "name": "success",
        "isNullable": false
      }
    ]
  }
};

// src/glue/glue.ts
var GLUE_MAGIC = new Uint8Array([71, 76, 85, 69]);
var GLUE_DTYPE_NULL = 0;
var GLUE_DTYPE_BOOL = 1;
var GLUE_DTYPE_INT = 2;
var GLUE_DTYPE_FLOAT = 3;
var GLUE_DTYPE_STRING = 4;
var GLUE_DTYPE_RAW = 5;
var GLUE_DTYPE_ARRAY_BOOL = 6;
var GLUE_DTYPE_ARRAY_INT = 7;
var GLUE_DTYPE_ARRAY_FLOAT = 8;
var GLUE_DTYPE_ARRAY_STRING = 9;
var GLUE_DTYPE_ARRAY_RAW = 10;
var TYPE_MAP = {
  str: GLUE_DTYPE_STRING,
  int: GLUE_DTYPE_INT,
  float: GLUE_DTYPE_FLOAT,
  bool: GLUE_DTYPE_BOOL,
  raw: GLUE_DTYPE_RAW,
  arr_str: GLUE_DTYPE_ARRAY_STRING,
  arr_int: GLUE_DTYPE_ARRAY_INT,
  arr_float: GLUE_DTYPE_ARRAY_FLOAT,
  arr_bool: GLUE_DTYPE_ARRAY_BOOL,
  arr_raw: GLUE_DTYPE_ARRAY_RAW,
  null: GLUE_DTYPE_NULL
};
function glueDeserialize(buf) {
  let offset = 0;
  const view = new DataView(buf.buffer);
  const readUint32 = () => {
    const value = view.getUint32(offset, true);
    offset += 4;
    return value;
  };
  const readInt32 = () => {
    const value = view.getInt32(offset, true);
    offset += 4;
    return value;
  };
  const readFloat = () => {
    const value = view.getFloat32(offset, true);
    offset += 4;
    return value;
  };
  const readBool = () => {
    return readUint32() !== 0;
  };
  const readString = (customLen) => {
    const length = customLen != null ? customLen : readUint32();
    const value = new TextDecoder().decode(buf.slice(offset, offset + length));
    offset += length;
    return value;
  };
  const readRaw = () => {
    const length = readUint32();
    const value = buf.slice(offset, offset + length);
    offset += length;
    return value;
  };
  const readArray = (readItem) => {
    const length = readUint32();
    const value = new Array(length);
    for (let i = 0; i < length; i++) {
      value[i] = readItem();
    }
    return value;
  };
  const readNull = () => null;
  const readField = (field) => {
    switch (field.type) {
      case "str":
        return readString();
      case "int":
        return readInt32();
      case "float":
        return readFloat();
      case "bool":
        return readBool();
      case "raw":
        return readRaw();
      case "arr_str":
        return readArray(readString);
      case "arr_int":
        return readArray(readInt32);
      case "arr_float":
        return readArray(readFloat);
      case "arr_bool":
        return readArray(readBool);
      case "arr_raw":
        return readArray(readRaw);
      case "null":
        return readNull();
    }
  };
  const magicValid = buf[0] === GLUE_MAGIC[0] && buf[1] === GLUE_MAGIC[1] && buf[2] === GLUE_MAGIC[2] && buf[3] === GLUE_MAGIC[3];
  offset += 4;
  if (!magicValid) {
    throw new Error("Invalid magic number");
  }
  const version = readUint32();
  if (version !== GLUE_VERSION) {
    throw new Error("Invalid version number");
  }
  const name = readString(8);
  const msgProto = GLUE_MESSAGE_PROTOTYPES[name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${name}`);
  }
  const output = { _name: name };
  for (const field of msgProto.fields) {
    const readType = readUint32();
    if (readType === GLUE_DTYPE_NULL) {
      if (!field.isNullable) {
        throw new Error(
          `${name}: Expect field ${field.name} to be non-nullable`
        );
      }
      output[field.name] = null;
      continue;
    }
    if (readType !== TYPE_MAP[field.type]) {
      throw new Error(
        `${name}: Expect field ${field.name} to have type ${field.type}`
      );
    }
    output[field.name] = readField(field);
  }
  return output;
}
function glueSerialize(msg) {
  const msgProto = GLUE_MESSAGE_PROTOTYPES[msg._name];
  if (!msgProto) {
    throw new Error(`Unknown message name: ${msg._name}`);
  }
  const bufs = [];
  const writeUint32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeInt32 = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setInt32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeFloat = (value) => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setFloat32(0, value, true);
    bufs.push(new Uint8Array(buf));
  };
  const writeBool = (value) => {
    writeUint32(value ? 1 : 0);
  };
  const writeString = (value) => {
    const utf8 = new TextEncoder().encode(value);
    writeUint32(utf8.byteLength);
    bufs.push(utf8);
  };
  const writeRaw = (value) => {
    writeUint32(value.byteLength);
    bufs.push(value);
  };
  const writeArray = (value, writeItem) => {
    writeUint32(value.length);
    for (const item of value) {
      writeItem(item);
    }
  };
  const writeNull = () => {
  };
  bufs.push(GLUE_MAGIC);
  writeUint32(GLUE_VERSION);
  {
    const utf8 = new TextEncoder().encode(msg._name);
    bufs.push(utf8);
  }
  for (const field of msgProto.fields) {
    const val = msg[field.name];
    if (!field.isNullable && (val === null || val === void 0)) {
      throw new Error(
        `${msg._name}: Expect field ${field.name} to be non-nullable`
      );
    }
    if (val === null || val === void 0) {
      writeUint32(GLUE_DTYPE_NULL);
      continue;
    }
    writeUint32(TYPE_MAP[field.type]);
    switch (field.type) {
      case "str":
        writeString(val);
        break;
      case "int":
        writeInt32(val);
        break;
      case "float":
        writeFloat(val);
        break;
      case "bool":
        writeBool(val);
        break;
      case "raw":
        writeRaw(val);
        break;
      case "arr_str":
        writeArray(val, writeString);
        break;
      case "arr_int":
        writeArray(val, writeInt32);
        break;
      case "arr_float":
        writeArray(val, writeFloat);
        break;
      case "arr_bool":
        writeArray(val, writeBool);
        break;
      case "arr_raw":
        writeArray(val, writeRaw);
        break;
      case "null":
        writeNull();
        break;
    }
  }
  const totalLength = bufs.reduce((acc, buf) => acc + buf.byteLength, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  for (const buf of bufs) {
    output.set(buf, offset);
    offset += buf.byteLength;
  }
  return output;
}

// src/wasm/source-map.ts
var WASM_SOURCE_MAP = {
  "default": "H4sIAAAAAAAAA+S9e5PcxpUv6AhLFGm+ySabbL7JJgmQFMV+kJZbFG1ZkjUeWbZGHtsxM3duRhaQVQU3XkQC1d2KGx27EbtfYP/cD3tj45yTmcgEEqgirdm4EfcfsivPQeKVyDx5Hr/flz/72c9u3v/Zz7689LOfrYtMRlVS1iJnkyZJ6yRn00qIM0UpKl4X1d1cHPxiNstSxidFVV9J04zNKl7OWVTktTis9/aiyZ005RlnWRGLlE24FHt7USV4LVgtcllUJ2Ud7+09f/78RlRkWZGzv8si39s7tn4F4WnGokPO6nlVHFyjv3maFhF0Iw4jUdZJka+RAC6xbTxL1xfHLMnK9KysqySfsWlRZby+E6VJqa4sLXgsqr09+F9dmbzluSF4EDHLiyq7qa6wFDNW8krC0frJPF07mJXNb5M8/qYqmvIHkQouxYlMZFFWXqInkhYzluS1qHKe0lXORM2q4kCeUz2DBo/j8yishJzzUrCd+H7vxPh66PKTWlQ8vTt45WlRcZZln0yKArTaV6MfAFzEvjh63eQymeUivps8f/78zTd5Wswznud7e/BGGJ8kbLHDtrbZi729CZdJhG/qdasGR+3tcSlFVbMkX/Aq4Xl9Cu8FrukM/pU1Kct4fY5usSgFvqYTsq5Sod4dqEDjpjUi2IKnDYykbtN5GgMij1nE62h+k25xXvKKZ3JvL2cim8RsLnjM9m+PPMiazz4Qh+X0Ir7IZjoVlXqL7uvYjs/hKMKHDeP98dInO00LXr+5TFc6a3gVs4q6dtp49LZJKjF2Bwt6iKWosqYW9BClgMddNrV7nbvxWSPOeSau4y8aE9OiOoAzisOS5/EDvAH8KvVw/vzz19GcV8+wFV7sOh6dSHyTyawpGokjd8d6ok1eJ6n5Mo4HJEF41fnuzSnPO81cjZdFIg7gmTMWH+U8SyIWcVmf1qMq4vVFeoYTMUtyGgP/vOLYxVvOePnsLo1de6DFQtZVcWRdxE5MU5+MeCqu9F4SvB+6rHxyVAu51tOoinrde+v/8Z/XvN2x/cU15wDrx9icIEseiSB87Pl+jnttQfjFe37qrejm4OwzneYXnTEO6wZOi/Pqseeo415bEJ6gyfsyTeyVYJJnguEov4ifoj2xXu49yen03sizqkQpeH2l85jzmie5fDR4W0lesqKpWRLLILzh/2AXbPaWD8j2Qbbp6b7XtLsokpg+TsZqWOkmPOV5JBif1qJiSQ7z7etWAd7NJRqH4kAtbGwrvmZNUHAa8+OhPXPFIkrwJbhNQXh/9FGUhQxCz7x1WMJa0EgRX/97kv+d7+1NmzxivJpJmh3LQt4b7Rme1gW8GxoEOJUNL9PRQkTr7C/5QZLH7Euept+LShY5T5P6aHiQvt1fjA0RXhdZEvWe6XZ8Z2xcNam41J80L2KTPWDXuv3C0nfVXmJaC+T2LG0Eq49KoQwr9/cHUSGnJ2VeVkleT1edBbuftWcWpIWmrnguy0KKDY8ZBW+z2dm25ku9AFWZxIs/URVNHk+ve4zMDG27B0sMR3hPv20/h4WI6qKikY+3Q5fPGJdgyLCDpJ4zmfwoOirDo4fXdf4clv+9vT9N/i6i+rdcitfU8GWRlXSC41YWhB8l9LBPzEQt8sUHvC6SAG/7QExmZcMyvm+ufwKjcgb2IRN5XR1dyziLKzZNecTAIgBLNOLRXAxPPHCJ+GHsL4Lw5FSdfd0z6cHXd4PWfB7tg300QaMGxuO8kPXt/hcpctlUgkVFk9cfpfS6yH4DE/uSM0eKPA7CNdUHjhYWFVnJK7HmruNVxY+CcNg+zQoBqwQNHJnMsiKJN4ZWvSC87PaOr/qDtJhNzxhDk23HJ6YHVVKLT+3Jc87lnNV8korXbYseM3HBKgEar+uqEcrc0Q9OvT4p6g9kkk8fqFdbJqVIkxxW1U5LEP567OOLRc0TmIXFoah6399M1EF4DsyvN2/uRkUu60fhV+9p0Ry3Mlyn9DuohGzSem8Pt0lgO6KsLvZFnvwoKlamPCu21SuyvumtWNnzXNaP6cMwe56vYUjv7R27DUH4bz69J8433Gro9yGyMuWRwFfAZFocsJLX847iWVlPEga7qK1XqbjnM8GdCeTVMkudFRXjlZpTcOTCadwBLbKyPgrCS/1RfndkOUAL+DcrTF5JntSDU5f+3mhTi/8H4ZrHNN9S+ys0bthufM03EcBHvZ7VGewt0lJUsCzNaEs8u9JfgKbT/IG6w2jOa5bJWWtP6pYg3Gx3E5WYiUPY7UVzuhEY1AxuxN0ClI2c48Xdd5pxZ1sX1i77SRB+NzKaGGtyeH4JT5MfRaw9BkXFoqI8woVVvVoQwHWcUc+lqhJRfVjzfD49O5s1UzaFqXpfjL5TUcogfGgb1Ektso6NDU1BeBofp8ijIhb9JQz3h7jXGlzCUOXJs7vTFFQeLd1ygsKbobUMdreetcz+nCbif1p3LkqZpPA3PoFn9kORNa9qnwAGmqdZ+Su8XYm3jQDL1iOL5kXil8AYrxPw/Pik3H8VeeG9Zp4f6ebH7QfAomK2SPHFdtuC8FItmWwmaJmrTf2GsyrCpkb9BVtf23Vg/r7a/TzJzr0hRbUQFau53N/bO7Z+tZu7wcmGFblgxfTSH/7wHfv2r+03FITgbImysruFTSSTB/xqv7USUdWzfnfjK7Y/B90M8yRXTgjcI7NJwuWtEafZ//j8An5sOGgXaMe6Poyt2F3u4VMIQnezVuAIDsKP4oKVTR0tVll58UqYqCrwab6bvl5TaCbJmzRFu3DTeem4HsCrh1tTVq4MwgcjL02P/pMMnmmd7Fxv8gTmCxaVNdiIM8mUMXXLORc9ADgZ7QVPKUsqbU4yJuczUUf9KSfJacLRRseQ1czYQcXbmbJOMsH0LmZa7myzumDTcuvV2P6JrI+H9ndWiZy7LiJsCsKLzruNm6wcW0/M1d0vK8EOZjLd2/u+EmVVREJKeFel+bVjW+VyDtMlS5PJ3p7drswFbccpD1qToa9L/5rgSnLdrMW8FpnI6709cSgi45abJim5MbetW3p01x272vfVWef9HrkDXD7hvYHOB3WVZHfUnKBsajwfDKU8psFqjOIF247HRl/Gq33wzW1pHfz8wdNSpmhqHPsFQXhVXYJMixpeEU6BQfiHf8gAJlOUkZ0ThDSk0QfvM2TgABiyY4OQFpAb7Tj5+rCs/gDn39vL0c30dJVNF0vkAQ9CbWuKmrNYLGBpwoso06RmOCKudxeBBHbBsDUYew10lUF4x/HWUJigFGCxvcWd8Vgo4n9cZays5xU4EMFLfMiaPC2i/bVuMzSeh1HEDuZJLdBreM1Zcqwf9SrvEz0Rq8+qrbqZVN2xcHfkMeASc2tAgcMkp12wE5gwGHweYPGq+XW6s9Hbxplv7b7fW7V/wKsZ7BHKQj4ZeQOvXwdaCncR2jayXleP+41BeGqhnTfX/X5DGYQX1bXpkybyHC6i+DDBajivdmi5Mv9HTVjYxX+sHkScyBIfVSxk1G5qneYgtB2J9AEseApfaqgurOKJFHEbkOu3XOncAvgJo5pMDFlMa5bxQ5zKTilrIG3oLzDur/d3JvBpVvzgUc9VWomsWIiOc/QFqg16PxZcbRLknNMB98dCSixLeRCeN6GWoqnLpl5rw1vYOJ1KUZ+k2A2Pe8bUTqzjgWnDRDW97sxy6j3gvd5zJPj4OxPPOj2Tms9mIrZe+gbu9FpHc1HsNyXNs7t0g/sL8j6xWC52rYhukZXgEFAxWvrbuHuNmjzKo3lV5Gii3fbMfkmWNeh+CcIz+knAV0lRlSjlWalDQNoR8dj+SNBP0flwsC0I0cj91z+y33/3/R/UrviUicw+dr1sPGY5zgXwe1pBKAHepKhoiMEe8M/R9gE9TFzVXF9FK1viq3AVt9RcWglZFrkUeCUYJ/MLgrAN34HxoUJLLE4yecUZBLFY4Hc/tvrJaC4y/ry7vc4kunUGttoeJzSshB+UxcH0xuBiWUlts6ONoPwMHn/CIqnvmb9YUdYwK7oNQXhG1lVdHLKt3Z2dTx86o0pF9uvu7/POviSJfQ7nCD0At2IhSinEPlvsMllUYHYUqXKOSdfDg+d4ZI+9Gt97LVicVGCA87JMycngDFt69GiDiaqGl4HJCRAfGI/J9x1R7+85QD8iqDzV8wJuaLQb8NjTGoTXlD+gKg60pdXkkk/Fb97ftpMRzB2Wr3YnvtIZLjyCZWJsNOdixuuOPxp1njoPXkiZ2JkldmsQXlERI1x/zUlOMjaFEVFctXZfkk2rImNNPf30BI7H+JIabJWgT5+n6ZXOd0xm+Kqx1a7jthVdp+nrgC9o8iqjjKYtecue2GKBV9vOapsw6OY8z2GZ5iU5FmXN8xjisEpylgaf2iKdTCQTbxueXlMeX5mgbSLyWtK3SzMPffdoplNGD73O8sh6r7vGiU9PuCqyEhzOZSlyEzhr/YiVyGO9j8nr1hO1s52Ku36/9RTiiiwvYm0FmvWrXbrKI7a/7kppT8grMbasL2hZP2VW5bt9Dw1+ozCN4Hd2CT2GshRRwlNawK5YE88ULe4Dvi/GclAqMd3mTV1ow8j+OyrSFKaZshI40mI6iQzCXzA05adJKs5SrL0SmDh147c03f3QpBCksH4FYX8uUdc1ox21dy5xVb6y7S2yQGia0AaXrFVX9grZEW70vRz4sUVl/XHPhtGOGJ4fHcxFJdC9Jw7rikf1ShtO6qY3KxnLlGVCSj4TW2gsJRmfCaZcCGDZ7e0d+wVBqMZgg5sNE0Knnxg+74Y2MdonDzgYCxAJVsu9zHCpUPZg2uCC8nNxWH5vTXdPfC/u77JYZpE4ik+9i9lM5GAAoDsVv1lKo6PJclbxLOPVGaa2lDDkPjqgbI6fF009sGupgvC0FY+45Quu4dcENqQ39DYT9XXfzj9KBa8uTItK8GjOwJiHK7/rWX/NNzblSXq3N/90Gm7hyuwsMDNRk1c9CM+i36lk0ygtJG4oRRaVR1P8HlYZhuKQYv8Mw3tVg68xMEfSazSyznA9V0s9cUDE5oSENXW6rvfqOCns7ZWFRDs3UO2QxgELXzs50qf0thESnpn8ecnjn6fF7GV3upR1JXimQij5tACTtdMUhGTE10XJ9mnoxskCR477OuEtk7/kZs+G1X9XYrajQslMJRHK1lIZkLT+RXRRZoJbe2zT1PqOtbFB/9vuFOWbhBWGnnLoW8eO+41BuO7dmgbhTXced2/gFMyzc3D4DoQHS55Ur8fCg0bh9tCsRauucvgviohP9vb0Huuy0woWQDG71Z+uYCzlNEHvYNZJJmcY8U8FThaVgAc+IMGVFKJqsBMwma/TpJKuCSEOy2u+zx8Gmvf7h7WwviIyiGt9iR4kExjU5oe9vtQFGnFjSzCP44/xNdJjnO5sMz2l+5p9C2kbpB1cSF2V/otvR9fAi3cVvn7nlXhaFXndkT4wi4BxUnVagvBOZ5o5dn4H4e3u7OH+PmfiPbyqWE5vXb6t1sygwL05vfbLtvNa7bI2PJu5SqAZdqP/EUP8AGehu61DneFOb0fd+5cp5YAsT492PfTPHV8TGFuMohPoznwdmHgFKAfhLfoywVvBonmT79O9qsf2jM6r5yQ1xvWz6zTfHrrIA5HM5nUwZAubNGG2z5L4UD4ZUjQTl1G93/XmgFtDwh8QNsD8vhtJXvfXyySv3wThReelg59ShahqUW7Qi8cXWLUp/mDVu66uSsywC5zUeBxXQko3bIoG1UTI2rO00GwBpz6tskh5PhN6MaBgQ12wMhGR0ONIffboXmsm9POiK+Nx/Gd3VuhYZka2zDJzFM/AJ1nhSGLptdZtL0UdQepnim6lkwyeQz3deQz+ebUOkscJd/SdtiAMRzMtM5Gx+dGkSmIMb4yFQrI5H9xy4edyux9ETvIYHAJs2qTp3e54og+oDQdcmSmHjl5SyKWnZ4v6DHr6c0gZEPGYf5vnR0F4wzF0wcOD1yVqUckr7gtFexKSe+B95CzLyqr4Oxqxd/xuZ3oeXO7PcAJZxfhDieSHLC4yNrAhseLNHVn5fucBhxeOuXc+4WnG0IUkk3raX+tMZUsyuNa5KqeV5+TvRZJf69qsOg/ohhNTVPsQ9W7+j1XuHcpP0iRm2q+50gPrHjQQlLqGc7nJlSNTEQb/n1HAmzgpWCZSdypoZUumAldxOG8y4mXdVOLZ8qhlXtBHGoT/Yu9Kn3heJEiWXJ+r+MRNFSd36u7envOhzCMwR8MVVqa3+C1101nnXGIsZaMbcoCpmVbHzU5IicZzglmwlXH5dZQo7nuzF1BjSSzyOpkmolpXwoiXkmFoDmdT2SauqOjXA2fRgU2nMVJlU5ZFVYv4UT94o1XASNFpitf669cRTo6DwW8w0U+IwzLbmn4oDsvt6ZedaH0poibldbIQLK74tO7H87sabvmIk4Tbzad2p9Bj+2cQntF7ZVjBNr/55i+/+07UfG/vm2//+rr91Qb+9xdX20K0vG5TYNch0aqGsUhRPbYQFbzWa28PRL4rDktWCayfgjD7j6Iqfnwv16v7VXSnycGvwlVc9dTuUT/JqW9Ze0A9r+bGPL+M0hmmVoCdBRuyx2mRz+7iPz0DzojeYH4EfH1kLMW85irLqeQVZFxSACXJ8RpVkBheqMgmIo6TfCZZUs/v2SF5svqgXgl2LLiXa0OHVmT22PwdhDq5rq0YMnVC60k2w/UdLG9YeliZpGlxsI4mxJzncUqJVhUliLp2KdwRWaNvG57XcDjEj95+yl6w3cNPyW5Msu2oSF27ExOjYXv20DMFtE5yLdixtsb0LClkrHwbHkkQjpk4kBQk7SzUu20BR8fj92o8LhWLsp5TnZbzo7/qmy90uObCVbllZUnKBnyI8HyjeZLGlcgpqDL9CFI6Jvn0y1UWa/MW+18K7Po+e8+4Cxx7m+IqWbmjoyoUMMZYsTjchhWXFrZYpDVnuZqBqXrLJ7nTWbjovbJKFBWGem/0lyCTVqZzvnGc1nWVTJparHe/AvXHhe4ncN0d+7Ewz42qqhYiYnFRw9blrMm/KNNGkrGdCZ6fb7fhWzEr58+7+7JEYuK/tzkIsR1CMQ0W2/Fof29PfRF1xZNavp6AK5m93WIv6B0cFNU+DqIT02nayHkwMvrx5eZNNoHH2M81tXYC0e7gimdWt77oxVA+3FA63JraOqhXTBPv7YGCKZh3krzeRLExd8CsSZMsqZ1p3E5InlU8T2qxiyE6mKtp0OGDtBwbMom+SGffF2kSHT27u0qC6ooRaZwG4GXd5GVpZ/o4P/XSjzWfJ3XM4CPGZJLHU/h/nibZR4xFhYynfSdY34/WcYK5Cpsj4SITJcLc53n1CxNg/PRuZxMCN6pfbiWmyeFVJyXO5IitmaUUbOxYTHmT1lfskCxmFkySWj4YeqDqOwBL8OmqhnQh62cr6mbJoZDP+r5ct6oOYukMStIOS3nVyacD0zPlRwIqdZc7laqifj6k529/3HUWwn7PxBj031g8BnoqLSkrtZMDwrsyKiodAPYq7C/uDNscGO2+0bc4zE70ZCIPMDHyGTi4lL8Pax0hdwrzcWYwh7m5uUuU3dL6M20OvDi6b2cjW2WDsGUuFqJKeTlspOyqvCpwcWzHJ3Wp6e1+7Jri1rSDGNxVwJbjEvyRiTjhTKcI6yIdK9DSbfnvOspi7SqiIp8mM2/kUpZL/WOO4k53nnfCQz5JEO5YFtIgmERnZ9OpeKy4FM9WSO0pK5i8k4W4NTDtkwvEntBhD7NTg5FU7os8wvV8sRVjpfsDS68son1Ro6Lguai1otllFaXIzzKY7UQex9N6uo14Fl9hzE3hWfybL7GzU5BnNJYV5DmKA9W5JpNyuM7AqPy67QIT/XU2Vifhv+0JvybaJILgsqwhQV+5csjwuUT7RnAq6xqdm5aHiXb0EBFThZJ33Szx7+3F5Is4DsKfl8XB4SpWKmAEFFOG3uaVfFD2AQP+p6tm4SFXk0reuWstPcVsNpGMzQqYdcB2LflM3GyHqpVEpOrqL0NJeiZS/GTrqoBI/4NB15LKjSx5ft2NUcwqIeIjTEe85Epgp7k5Mt+bP693lXBBFmkqPQVLpSn77NvWskgXAnInlTsJ/Yd6ToXdnrpC7cyFlSb0LRXH/cbWl2SVYib5LPS6oWBC4NWR44va8Gqiv/qJEmEGnW1dHXdagvCa3hykxYSnliHn+K1qPDfWpmzOlPXXKz091pIgfOxZhDJRgXGl1yIcoLQPTiHslEOj8vHTKpRuY8HGeStjDD5AazcBQb2Dv/V9Yo4zjCkwDkyVXVk3CPs5srCcbfRaVUyplLd8yyDYNw0kBAWOlJbPqD7UwFGUswRL6zmFU6HqHC/q3BQTcuhmx4JNe8fZTXSdc0GoU3VyWkp1jMfT2mY4ZCIrYFsmxVtWZc66hTEQiAi4eS+m+alnkTMbKY1GERNy04O+LszcR2wiTPnOuhsj29ujdMpPH9kxFe2yKSQVc7Z+o4f90AutWzgmNRjS522Il/b3GI/t2FoY/hW80gVkphD4N0O7plXTZz5fadvV1uq5+66rVO5vvJ3PMGBt11LDLinHhLGSLh2/QLj0G3Z2iUr+12/yBGMHPIHN17QWaVpcAmPgi5iXtUG3+uggquoim3iTEJQ7ajgJwVJ4aIfEFXAI4/Vr9QTLCneMH8kjCabgGu3AShlrJ0uVigtmW6aszdO0cmNI5ucyyT9Rnc2aJMbSihoGSJuG4LYH4ZPh+jjtttT5uje9qhQLEifINfXI2GnW8mvKh8xm6LIVsMWM0GkSXellAMCK+LDbajIM26Yg/Eu3yRNaQukqoSWjuN6xI2aCFuAL2D5J6ozc9mg4+DOQWSqut84yBrMa7Hlp6rnRTzcyMfZ7btIRPmpXQy3aZGmoWBdc4mXXrMCk++v2FtdJZ3RRj+C7R9skCLu2BISY9heqI0hrNMs+/DBbUPP1a/OCkpCLUqqQN6dvC1H37LSMx0NreqdhzVq3YYbngDHg3/F9C27pcxr8AQYo12lUaeNDe5MCEojdJdCs7hmgm4hqOr0eJ9MpJRHt7R23P1wvtEqwMVaWmILRoXOUccmDUh8sc5DhiF9OhYNV4vcYolIpxP6NdmGUAnO5GMwdZZHktWclsrdmmM1+3ZfLh8kFOl1MJwJYv04w3NmeY0wCUBGFAaeU9ilFfQmhZuDz2HqlSolOw2YMEDWirLTwBL+iSXEIlgE2aR5Yhvv2vFoJCHuJTvbSNRMboxAleHRxvfxI7WluYDpIm9bV/grCUG+FwJNionj9RigI0Y20zMNHg+va3Z4Atl20UiAi36OOwjQRuI0Ao878HYSXldrbRhD61FRUD7srsJARYAM4qfbyfMVzyHPQJsdVPQXjqjmpCh4DZs39kZ3dX3IO/upHY9u0thr7+38YNK9jgW/YSCztrAwT8NVeBVuS1zvbD1zMngkWG6qaF918vV87oeJsz4fDGKqQS0RNBbUfmElwxZ5jsbAR7OnrzvxKg8/ZSS1LL6uK+qo96eoFWux/MuDTO/a29+sRzU3D8Lk17AKU4q3eLME+yeDeNnn0L7odHuYTn2+6LvaXrLmuYnfXSIm5FIjcX9CCATXiV0xwxtTDTne2Vbpkkl9R+Atbrwh/YWcb9rm0tYrB2Yiz+hk7cONNupc66X7Nty+62mtEZLDQaY7KhsK84F+3E+0AEdBfNEMJ4F5IIlicVw4Y7bJvOwGj68qUxKoDNZNRZdGmR9JrumLmGF3XBUPpmceZCcOU5eAkaYuK0Auz1tnd4cc2tqrNJvn0ubPdWxpk8oOIfv75JwN7vGNvexB+7CsA6aqr5iB8vsTzqarFtO/TV4OBe5u9vf0Eqs6d8jCswTB3clEhwyEy4CFMdv9t5ay3BEExAempO/NaSBqu6BRjqZxXdbLzEWOpqKfbj9w8FSz7wC1cEoM5Co74KolV6UtWrsP6/WeEFPmuiJtUO1qvtq5XlZyOo7y/02qTNQd2Wq7CgJvVROWG3axG5XXbBa1INCN5tstwJR0f6zdDe91+/QHCsLB0mm2riBevMHcTKozkM2fHWDRgMBYVgDPN4VpcQOoTTQ5QNmdbU6eRc0gKSnk+a+CU8iibFCnOISq+J/eTUm00tedTuTstjzC47NMk37/ZCQQ6P693hFjth3uxm67NcoTWTF3Q76edTWWGY+rY0zq0VzXOHbDDyqqIm0jED11VhXzRLSD/QE6q/cvqLTB8N2TynsSF+s919XjEHPpDMUsinn4B3+o5srcBIQ9mwW9/ggQNZcbFv/9pPuonQTiaNjJikWEVk5yDVXDwN7VFVuVZ4CPw7rVrUa2211aKvxn8MlK+4KyZQwgkBRgZkwxk/wzCZ1ZuqVpxEGAIXPbTJK1FNeH5PmwNq+QQ8DC3x7O9MTkUIVSC8OEKwDdB+KS1FzOeTXgv5aVtPQuNZrOtd/ICbS01hpN8ds8bOtAGAUw711wNRF4kj4Q2O4UdJCC4YRnalqpODvQGx2841quNemwQfsB+hdiDEeuGIHzQsXABGah9gJRb2rWCXR0MDau9VVfTxeDAGIjkCL2xQjB+AXb1CpUgCyzvGC2fXqxUJ+LtSCV04tzFZzoJyrhcwG+Pxqky4JVc1ZiySqCrNOJ5jKuuRvOBPiGZvMZ8wg1voiEWq12y9xXg/TTYNfZWgwR3R/YHGGE4t59kCdvfYRRv9QBxJuXdTkiHYG2sZOLHHYWhhLtNK30ACmLgahnT0ECYlPnozT1Ta6cQlI7dhiC8jNsKtCTQ4Q637eYtNzvbbkOys33RaQDTx4J+y+SwL2r3rPZF4Yd6QSH21KIqCzAhz5saL7LmFYZ/k8kmu9qJCqlqr2seLxamzN0ZBMgh8Dy36pRSMdviQ3tLA4/nXq8VAJLtzUy/pAhi5lCHBnlOqqRoBhmw52Yiy/guU46XD6dZzZo7GoSjk15LxBBB+L851uaaA3asYh+XzWZsX4DjGXZxHw/mzpZVMZGMFmn97L8f0H5vtNhbnc0dzspmfno0sstTaYow1SxVg7vvwCXKGk3VvT0JeX94tZTyvGFrkUsWTcEmFaE12oYTXiDF3r+fvPe5i70Jgz8I/6OFgMBZuq4SwAz2pvVYSBLjaT1K8d+9+1Rv11Zh7HjXSvFz355W0uLVBWPrioNQl4crv4GkxUalQVx1hbCbAOEV3LFaTxo3kVYxHzomYVKNIUL3C4a7ymlaHJwgOI+zqu4KdkH1dB2APQ75JFlsgUuDx3/5w9e/3dr+dMiX/C+NaDy+5N+9c5gTl4+3TVELRDdJwQEhP1vSjbuFwVpvzPxOi1q+UGYW3CNEPJ/Q9343eBIGCs4ItoVPQv3/y6GTjaI3PbVR3lR6BK6CuAdw8+Jvou50Shf8espTKZ7drSt8gt+YbSgWQ1jwcp7yfMwRg3nlbdMVX9MuWWUyyWZCuzG1Qf1RWabNdJrx3JeVio9UV3K4JXLH1q8gvO99DdF+WZP3eNJMN70qmKkLEyG9qktSzCAFnRZ6iMfKz98V1/G4/RGEG2o3SyZRfWTlXG0MVEiXhXy+8uYJdX5lbZUIqAJzU2Q9NfgtXmEQfmwJnRMdzBNZYmK+WS+C8KNsUtXFQXSjjYD+YYcesIQoNPjBb/TygVuApscWQJNxpOlsLivIeoKYGD6ETd30RFrMtsqpxWVCjyyBkCH4LRsphqt3K0mAFMOZXYotjCXxMByqWy0Me8OHdrAW0sLMdqltMh561QQQnuAMhkIicVj3hDNLeN0W6mAw5nNttJvLrv3vT02DiWjXl49NcVrcsWomi8pGzd50DoK/FYYPbnRVHpCL8yOqdicNSvg0Nd8OxsjhU5R7e/pHJfh+XBzA52CqZtvSXZVjvnxPCc+pkeaSBzQxdG62Po6SQjVmGNm86VcQbwHj5cagMCqHD+Rx3N3FkrD/ILq7WNLzNC7fQM8j/wa6DSW5wGHdKnb0YLSATR78sJbvCBzXfWYoglinLL9fqL0o8CuqkBbGI7r5jO4XAa7cq2kia/Cp7pjvBWbrR25VfYN5cS34PP0OwvudrStCONAkAr6bmBX5A43kCkDyGezjeT2H1YVi9eAVk+tdHZVg6GZA2pkO3ZbL3QxIIfb9ZbzwBVjJk1rJc1UUtJTJQpxP3m4fSrI7YHG7DHvqjuELafpmM+CWMRrj38TYz6igNibDP22BbhCQoQd/Y1qD8IJTZihrlaRRV8lJDS9+xQDCqgm3kXWRnVStsw0rhxKsRISoI52LZlPdQPiaFSX1hWOa6q4JwEtB9DW44VWAfYW8599H476Xshuu++KGuCP613fI3FxZ9XVfs119dMgRKwmQ3MZuCMJND9YUTpn2Vn69t5WnScet3oQnRcFH+9gg7KEQ1ZpP6U5fQgiBySwHI+qS7SUgpCyNRTLL+OEHENw4D9FAhqUkEZi7J3H4RvWhD0owqR97gprHvbYgvNPfJTp4bpeM2Y/Xti+O5Gnti5nubH+ryzuqBICgilh492R1lSzZk73BFKFeMBVu0tAZKqyHuYj2MatHLbLxtF6GuK1CVHAN9zqbZ3XTGLSBWno++83SAvklCvc6chOzNWHGp66G9jJ7WletTnF3LRec7bkUtQaniYXeamz497XwwJ/4RJ18YNX6mSeUGxdRQ6QHxyPSIPzXTpT3J9nNr9udWryAV/37dP0i+LxgUVFBJGi/vW67NQive3VzTAPwlOvMBEIjxEL9hfaEFPWLUeXtA74A5UYSBOBiK76He2BcKTQwNPxvFeDv2Gm+NkqjTurqtQXhScYgbsjjD7Ho8hRjb/ELry5YfgiMTG/Y/gVcb2tRZUnOa3GfMEVjEVVYkNyu4JD2hykhH2IN5/W/ffP9X36fA8ZsJH6PG8q/8aT+Ij+6DE4JLdHcyAfRYNoxvoF4CfaZUthpO5AEOeUQm3j5MT+zA89DXkAg5IsM0YtxAr7poswPMxCiK7pz6sdD3ozKRrKalTtDeiNEbV8MH2PvhEqR87ROhGwd5Doi5EC6jhLm+W5RiQcTwR1/yAVll0KqEJooyxwtaMHq0+OwcxwtRnyC6oyvW3ghVCCuEUM+AoNxPxXryH2RCo4VLS0Hxh1yFmP5GZadOL+DcNODt9VrOqPMUgwinW3hogXX6Mbw9S14eo5+zZPZHJKuL6mfMbqX8cxKY5LUeNc/l/Pq53Kerss53375ijZpDDiHqQD8rGpvSnBLdxMX9oUoCXnxlVXXrC89grymaZLz1LhmAE6FFTzBPXEdhN9bh7mzeStYMpu7ijcdTxblENZJRgn8gSNU6f0qEdD61fq8evj6UDCNFys/Niow/DPR8Z05zUF4069N5tWttqgBzRUKd9ITD8KdQW/q8YAkCE8h2M2bN4/MLXtqFvDNmFvsJpeQIppLJVTwgRtJOw/8ySUpb/JoTkWQOL/Cc/oQ3U1zG3znmR26Mm123ZEBfDBSnIidT7r3M8nrR+E9NfTUo9RXbrkGr5loI6DEJW93WU7MSldcPhs11XzcFsn6qjQQUgIlB3wRhCvlsgxg2e8npTVtBOFt54ngx7i3p32dQXi16+IkS6xfGQLxpy5SD6xR56kyA4PUyMjq1FnhI7kArqVyB2ECKIeiLd7AZdQkt1wxbsuYyaO8hhK1pLrVOjO7tR4VP+gUe9Czx93Tebv6tBQ5YqiA904WTRWJbmeXM840bqfepF2CtsxpgruhLbxq+ACe6cfuEkhuyDjn/exnFuf8lyNZ0seDsiD8ZNiLegBkdURQNU+mNSVnPV9dHz7RYSetSvKqnqyQlqPoqIZzfSzVWHJ0p/myeKyWbQP5oBXTaeaowe9YwBw7neZrViUP01EKt/Tn2PplfGL90p/kR3Gl9eYq6xiuv5MGhMMY98+dEmP4bDq60ZwnOR6x7hPwOL5ot0Mdu64wAvAPgQXoxYF87M8p6vl/P14t9ygqskmSu4y1NpUSbFKxmPZWTwNy8LTUoWJSrGm041kVIETw3AEniVIx9dyWTvPPeNnmSWW8DMJHjje7LRlo3Zf4Wu741QwXw7YjV4GFJD50Hbg520dsi1wcYP3C6ocE4Wc+9XZdHpEabH+PIxusqQ2vFKMTj3ye5oofdOBSl6uhP3oFv/VAd32PeSzHIVwXK9VvYJ5ZJwSgPdb4AK50ZXgjN7utRJ1DqXDK654UyrAF3xo418HAlSYvr0sxsGCzt5yc7h+7eWQixV3rmzcqscSyQpK81tOR1gaTHNJicOdwwU4Xg09ozW7AJa3iB079Svv3vX6umT5C0ejcHKhwIVIDOyMNJjCnAdbER76ctX7bBkYLLOCQNmKwYQUSWksDRWsocnmwpMs7qvcH2qBY7wQWUpHP6nkQBl6ffuvON7lIXRQ0BYmAsH+fKRkkE6HnGN6pyzI3KLqqjoUIRYurJnsYairnTcdRJknugjaYGAcRGyRvt4kmGjCnWQ7hw0nRVM+8yXtcWsl7dzHXHYzIiz2Pj5uPN93ZXjMNugjmqATkBbeRGOYutNzV1H6+pWVAZNWWZARII9qMPCAaKcobDhi3Kp1iOd6upnxtS47cAh6L/u22J/mOcilx/bnmEVOZzAB+Yk5HS1+/ZC56+gW8cG8dEwhwH+268tGqUDirLtmPtquyEl49lkv10wEVxqtkRXnLE0gg7HLwUSqffwxrFYVh+EReUzWNBNmsrPk8F9VlnBwgaGmAEeQ1bLOBWCqBFujJGMFgDo8uxCloH8J6hvGHswD1U0AkDfYXF8zuszxikIu/53HK9yucYL3T2FzqzxaZYvxQovTqO/Ct4t6jel60hK9LIf806/MYDTbt54gy+bbXGW989Te9Ysxjbeuk1Oa3k8PpOuXbZLEuh/2+OGqFtrPcZCV97hFaFVY+R7sRtwmDXtAOArVr8nqp1gQ2//f6WmbWp9I+B08EHr5V0K0oxvDxQ0qW/GMnCMCgqvw9IwFvWgCytnJML0XGTbXd+ohNOb4avEOSjQGwrunO9i0/IUiUIp+vjW6iqGZz8BF5iEqxbnY7fuTHkod4SwH5uohrEoT3UE3kUQv44jQA1gs0RPWhlkX1IcRcTLXWk1UAUfSCDJBhgiZIcVh2ar5+eK9Of/ga6lFTVZr1k3T59Q9fW2typ8uLdp4TOUR0MbPOXL6map5lCYkPkP1QLLBs/5ISTFMu59v4af2zH1PGBfuDSyOoMl6rFG0RZbwTNhjoqn+Xy7t6YaPCqLSF9Ahjt8KLEPPcPgBQswq02Rhwp4COugr4F3lgPHM6xjzV9AhOyodDNHXH7Y8gfOpo5Tw9+lHQlACK9u8g/IixBtJfz7E2K5ZF5XqXtbqujsAJBAmz05QpRNEUrPNTjE1xz3C4a0ezGDovKKBOHBMKcQxxgNhEwNwSy/qMfdBLSqP9QSDAj0LM+ROak3Dt3vYgPHkAKX55k504iGQq8sdqpDiDHMyz163AX9BJpwcyo8GCTldliJoBXCtLqBm0ylAXMCcu6UKrDEbzrFxwfzRPKQzSElrhwCFawn5AMNOfghWdcwOCGS89x6DvGu9s5BirhNYEDV+3O1tdXKhJ3FyJ9+B2e9U92JUMhCAZM0eOhiDvOmxRRVXvdsmQxuOIjusOlko3NVKtnRkvS4MU9+bNcBr3aFHFpn2tSV5XBVxw53rZysR7y4pSYV6cCQvgPPXkh9dVI57dhSxxEJ41ZeEAR3Ne043VRVociOqy/k3sULiRueJQXmpm2TXdipE8xVR22cJNh/SwVPCp3ZbxfQHz4jkT7ETnwfmWIhB3VrctykCTW8mAApzDyodV1IhDpKrrVPta2x4X2jVytZPHTnEbeQJ2hTvbf+0FQl07r1/pPmDnuYobDn6NA19D4PGHmlcGTMvrClssz1gnRrPWSmaiVm/bZpudiKu+evGtVx/IZFqfUaFVjMNtdiKrBjOY4Q3NC7CHr3SUaCb5wWn1R1GXIou4ii+XRHMRxbBjJVuIPjbRjuQL4ZYXEHcwfaDdgyDtXAFEXXJAgzDmGnajqEXpibgWZRvAVXX0MI7zpoRkBQzPQ29b3ehpoYEx/YK2034kdr5dUFmLv1hCp+HjG3qzQqAVgl3gR+41BuEj7+EKhLvNdNOPirZUOqun3whVPFSX0R042GhZpYMDBxXfXKhS5EwB9xXODucqLCYXjIb5FagmQBQvhZOA33+fbwXSWlORPxsFx+uEXa86Md367TZ7AdlE3eYtar7iNCPadb/1FfvW0/pyoHXL2+rrd9fbw663h11vDzveHrYHWj0PIl9My91+c3aomi+5JjJ8rmtOnZkKja+1eA+Y+4Jb+jFMrD9DvDIIx3AifhApJp1xKJof0fsOYbg3TRY4eFILLIKeNDNdE5UnZSnqO4NKMSYVnlLsTnmt/pqLw1+oVPE5r85rt87dWKSiFjfMhGd/RjTkfzuK9gWlxpjL3zcj9Tn+4z9//b5dqM9zJQSMZYYLvsnf/QO5DDCNq3rh8idnlzse0w7C//u9yFpUpsmo3TfC9MK7qDb/13t19N7nbweQdR1vfvOeICbG+PjiPTtoL+er9+xBrYC05v3qPTvh9UMniaaBtIPjblMQrjtN6P5Gd91jNympOJDdw6Gtm6rTktgoQMtzSqw862cdyj9PHg/MuI97zR4wV6krHXtAf9DFFZX0U+tINFpt99xUIN2radD0sm002/6pT0iuSehbIaRD3PTVWK3ksOxBT0RRRMWFSHw1PR1f9WUQ3sySPAGYr2ynLQPRwcX7mZDwCKQiMqwymgQh2qf+PpVxdoC8B5DqBHlOcdVOMRhbuzUMYCu3XlkZUgbBVtZxUnQEVNkJgjYNKgb6JCq4wA6tWlDsB9Bs9cleuQD9CskRkPkVaijjU/DK6fpGVWa23SshrQuAva2KwyRDDjdTUQoS2D328SKbJK8/fTqek4UOFe0ShGrTrRebbn5UM8va3CnzAkJXSZaWEiAqwpCYJBAOCkdSsUALdrikeb8PW9ltsVCLjLHcbQrCX3cLZTEhKjcgSsej8iBckgYl50VVw1YxXIamaVqfLNNsMXHWDI4Segloh3vL44RpKz4etC69QZK+NV/N7A03PwoSoVBjwnPZwVzGuMctt81EZSkOqjKzSlFN9Txk8oyE9eydljabqNVRCAJA6A4Zv/C/KTcmf9MhAuJ6kpZ6OVs7WeHT2+zqbS8r7NUurpT/eIRfEULhOolUCi7IpFxVQgLilINNpVi8GF5Wt0w4cDTT7CUWNXuuzEkRs5PAcKVbNRMMZqDtVXRhUsqx+ANejXOVlDzm9gxTDHxSGmqLcmKPrV/dTC4rOczffm9IHfMewCnx2NGgEuvjXpspsnb0lFF+0yMypElWxhNhgyGuWSFhqlv3yXgc3xkps4YhtLx8GfhsIBNga1QTE4lAzYpuDxxhJ451iLx+u/yIZRo7y7voQQE/9Byj8qHwUHCqgFPWm3CmkBfyafHfewqaXMbHedw665dwHmvFO73u3YfXk+Pbg6RalsTSYM93YNK6eW+UOGby3tRBLZoXfnMa/kxnr2GeBc3+asaaGY7kWMOmUe1MUs/v9lVazzso+GlcpXh72ytA2AU47oqd+0a30+xsO4lyZJbsbAfh/YE0OAuC4N6oCpp4QxqIRPBsQOpt3hxQxseEnk2epkNXRBTGoHG7W8dvEp2qhTA5pR7aIpDf6Ehto2bdzvEDfyIMv7LQ+HhtO4yV+74kQUq01Oe6PaLCYz0xucQGx/3GlsXUpAe2VU/VHUe2f8CrGYATALG3KKuW9ZfkWAK0twf+xQVPRV7f9Ik57ArqILznCJHFr3Z/aY1K1PMKQh9W2uDBHBzhXWg+RTCQd2mbtr16KWGlQq0+FWDah1zQ9II6z+eibgC0I7z/RyYVMcMKS5Ok6KYcaugGJHWinNlKcFkg+ZKYTouqftASwEY8jRhx6qg3HRPu7mMDU4DGRAe6QLcF4Y5pU/CyEFPoaLuSIPzMSOj6dA1x97i+NAjPKRh0NXR6iY+QPvfUpqaSCLnM0mSiM5zyxXZsnM7PV9CND4z6L211TAKhzQBGLAZliO9gE+luvaJUyYXCDUAAf2rRgULw3YJtToryIIH0yyjlWalakrQxTNrYgolDR7yCAAZwVSeZvNQK4GOHFYs46BVsAe70yzrJuGmC5IeLSqc9g0UauRUTOkJOFc0IZULLBZ0NK0KlgvOymb2AA/syZYi2D0cc1hddMIpC1lcMEAXC1mMu6dartnUybTNMXebh+MAA2tdsK76if0Tg1wXEPrB3b6pWStbUmJFZk0JezTWvMG5KN7NUvWZMQMFolJcyBR7DduwVzVB02yfCC8Uc0k4abJs/6qMRlz0a8aserVwc3B/IodV8ZsmP4pFHRf/SNGZTUT0eoHmmv9ot34MBPZxHVPOVrg5+yxu9VsMx4KYCY+GLk4brdkh7ojT1UgrAKvhiVfY2phhsbo8fQEMdU2vRtw/yD6cHSSw+nFbisDwxRcCzGxZqh+aV0Cm5V0RGsVocYZqe7w5+KpblY0YbestU7q8t3ycj8rYlQNPKPW69K1aHrVE7UR6XCvDwOlk1Gk7MEt20Uo91erjZcFy1hGQSgcUtr3SbQVf2lOnU9wgFvp9lS+1vgvAsJDk3dcS2drd/+eIXMe155VH2vzmG6slYRClM8C1uKn4WmJfaArwAAlUJ2wPTtE1reHEgNd4n4bxQpsLmMC4LVTsDMssy8JZ6Vo9RQKncbzTYfCTu7U726ecvu1LL6z0kCsIb9jNr8WCQsmLdllE7Hv3LQQCYHoiLJQvCzZHjzI48cJVUvrmnUedrUzo91d91m9q0cvLDqOPrgmDqnlof0zC2DEERP7EEdzuMS5345aybWqGpEK2q8SEVozBeH444wGvOXDCtCjT5VSNuh7WHmHyFx+iXCq/pvEJEgVLnx+IOL8qs9W5uOwoGNIfy0kp5wy8mND0nb167tj2NngR7/JiLqaZH+W9Ogr2qje/REbwngM4Vp3P9vWx1WzMkMetUHrSCFliqXwUA1bPJ4d1hOZF23htUKMoaM7Z+1asdEClQzIiqi9vrCgMzcKzAVy4+jNJCiguYLC+b6TQ5pPpeT/K/vcCBvQhWyB2PHk5xqh7xnr9SIBYmfn+9y4VqJBbNKTvikoMbGp1GEGnRSKFyHFmI/qLDlH27vRJuERZbZ5g7VhwEK6AXyZzvi89GqcmPR6QwdlophACjMlvsslfKKwhITK1cBReV5J4l0W5qdFqomiierfcLLDA78iq1C1WJpkfbKWxGRIW2oEL/dQ7c2mRklDyXm72iBczndlMJ3q8CA2xPXiNojK9c4q/v1emkAiO2Hi6Z+Jf361breEpF/vhePfKDfb3QuN39+b3LRIZLT758nz47fZx1ALwG6If7pSJe+uHv37PQhMr8ZDORnQ5/+Ic6zAtPlwPXuLSC5V2vccUOvdf4ELv0xerQMSuOEPfz4QDYmpv1vVQL4Q5vDmghpcH1CU8AFTArxA5zhsuLJWhuvabdkVIa5N5DrpQ0ZUmssbqD8IXvIEW1imi0PD3gR5IRXWkchI/hgBYYDdqtEqEnWCMEd32OV4JK58TbhqeXGVtMqcDZlIOfgXqDJo/qaTzdPgVAdLhhPc1YnTKC6DqFf8Nfp62in5MM09unO2cYbF+nmCGcnoIfZR6VR1ZFUHWATIDU20avva6Oqhj+WmtFgJ9Bqbfn2sa/F0l+mrE4ha2wqKoe2v6fCW3/tVtX1Csp0lnDsNFOIpLwSQH1U7L+5YoHT6EC06pZepdiJnOywD1IuyoQSwcLZ6YqQ+Y8XElB/kJ40ncAxI946gjC7wvAhP+esrC/4+WZAxX5gIzMm1CDBFQD4k/5n5tJltS1iP9WVPtfFbm42WW/sxnyNiwi8t/n0+J3lRDfQSipkkOUBnC4h9JgSP33apruqP+H8u4pSgCdk011NoARocqq8AmYn9/x8rsiFuMblvBzUv6SDMbvlV/5K8UUg2j4g7Ig/Hzsyvb2jsfEQfhEiaHSG6dLIKgBVPN+66mDSCLlu7yn1gO1BoJzUtcKHVA91l9dDYcrsL+YoHcT8eIWokrtAiCt/ITSHz9aSMg4na6pLHh9mTha+kVk9DevkqEiMldhgB9RNhOWDRaRuSoDXThkD/4ujMpQOZ0dbx4op9Mqg9Vw1oZvqBpOqXi66K2v/S4cQ+fztguI3cojWYsM68kgmMbaNqJ56PgJhpgm+6MHmSbdwsh+pVsX33O00u2JU+l2wEtNa18WslND1iuKe9lRWOtmVkEZ7AhKp50dVR+VGB9EJ4HD5w1VRawpO+fa7KlAwVZH6aGL0IcPBv+xHCZPwu+HrnCIVMPZUfUoO4PwqX1t06JiwPXOVE2Jfv6xeIuVxOMEn2gjfbFy8V5L3+JmDW9hD97YDobHpmndGm346MIVjiCHGA0LnNc09jkEb9BpgLbk/TEF6uOrRVLVDU/v1pABfFcbWPQ1FOSh732Vx444CEc7ScY7SXQnpxbThMy0ywDbj3dPjlvw1rxpHxKkNer3ljeZApgnKRVJVjztPNR/8uC3UtWjom3G102wOMV+U5IRA4GgznzzsRcIFmDRKrPXpqUXZyev9goXjUb+l96jR5FvHY4fGAKXPX28tkGm3uGi8JZ2nIPtXSrWlBJGR+eY551jOvETVzkI73rqUhWlAOPTrBCPrSpTSiGtRAbGJbzMFBMOYNchb1h6jMBbIQyO/pWrVtmpFBBXwzn8otVcCcA6vOS0RGA9PLKaoPg0hRA1ph5iPDtuIqy5v2OptRErxLfFYtl73spXAEQqpuQTumtrCEi51jjBTBwmspb25VKCFdwzhsijOSawKYwXzH+6ajMS52Sh8VqcgkxoBFFo62ftEgR5zjRnyaGIb3YwiJ2f9z0AxCoaoz2ha27BLGXTX3UajcP7FJELJ7mmojKUwbiPVOWxJU4kuwqfuGhqNksguTcWV6gp4/EOWA+SEh3uUOvfS4zrUmUzbjSgOpLF0fqAfENV4ybT1i1WVPgxKpGyGUuEPIXHXPGDywoj2ebQuqHaIBQJJRVw8wpHVJ6GuAyrRN3k4leyKCueF966xeNhIVA2jxypbgqboRJ1qep6p1ZY54vc6rRjTk9TYnFB0WOmbnPBr3YkCq/my6FKYZUj3hadDmkE4a+W9tGD5MGhGYSfrnxkB3f610sKnAcvGuVtKTDI8YknFJ2BvF2cL4LwsVXYjN+VNN+VisCkHPTWnAJodb0bTqPDx+bqY0V+EN4Zhr1m9ay9XpJTbnNd0JaKTGipi8spK1HDQj80rYTbJABITuUTwvOg9ttDeNlwkiC8txROuwXcdvCxaWhTkZUeuKqk+9j+2VaLKyl0rEcsPW0zYvvNgkPOoGomR59KoMyKmKcJoNvr6KOCCQf3Rj6z570hBWNnbtgKgGKG/jh0Qm4PhTbNq+9JHncFJnJN5KaqaF3eWaL3YFAOsdwq4Xktn78bn+OtUfUXg1InuIosPDEkRAxX3cdbLOUTkcp7Yyq4UPsZChEGvUaiJ/jC/cX7FbArQRB0ypM09KqURdlg5iEtrEgf6z+jqQSktKc7XiWMAeI4fOCVuxSYt706qka7/fZ6qPEmn+FmV4Mc42kBn+5GV0j1hA4SgQ3ExnpTtR4OmlLI/onfLZpyijAP0qY+AmeOOCwvYRkx+n+rI/YW7AcQpZM8vaZyELF4vy7o/102aQVQvw8C+P9TW/BSCV52BbtKsNsXvFCCzjm2lGBLCa6qT0+VGepHcVHZuvDFL5CA6bNRcIPjEWkQ9tALfMgBCulg3Yd1X4np9U77Djs8JNqM80YC2ofynPV7hzk/t5m87MACwJ6Ap9fLqsmFhZKid/Pyd23lKDkMVFHoRPkSobq+erMkj+NReFqFBmCZsqAExGHJ8/g7HlWF/EHztv1ehQ8sve8t98Te3rH9MwgfDumpP/9MRXMjwAROHz3p1xDfQPj+u36t9u+PyqIEuNrPxwkIjsfEQfho/GjaqRX543G1EiIvUG6icBFgm32lhUhQAQT4c32AQ+sP/yAswXErC8I//STgTLjPShYiPRqtUdf9DYERaNiB1yt3oslsrR9vVjkYdknFlDzg7q+VQB/8UA2UvR4LzLALwlEEi/GejrE5CP/1Jwd82BdH//mTdwpLv8Z9/W//Jb0rsNjvV+k8QQRFwAMbglL4GPws6L358h/rEKFjP1ulj7guuDY5qkQ22wzJXFZFlehiVLwzqsQwyMV379mDnbXY/v3/vldvT1ZyMzujZmCwKC5QSp10hf/P/0qXBskJj978n+8FxrEqsEkPigRzZdwg0OjgHcMUabLyfRFNCI8E/h2dbEd6UDTQkjW5wvFZddy5Pf0vNe7+f7k0Gnehjw/puN84wJyUG7L7M0qMHL83eiTyZkf/+T0HqKXSiCOmwcczD3B3RCflFfVbYbO37qWrT+Lfd9sfObHXVupAQOEc2xHetgJ09BwUGD79uOcXkyexAsfgm6EIn6xFubOAFSebseZT2ErD/mACbn2otCYkx4+HDk9TvuCsmcfE3kLtv1pBG3MoUyiLSBSuKbg7rtpwOIskFgXulG/7UXLg2uHnnQGxrsR+bMsdoBpCuJwB2n/C5VqHZgudc3RJhOKt3i5SIPSaN1zcHfvXJdooUbgZLytes/ZOAHeCXYRjCDwmJQgAWvqgO5YmwusF4ed9YJ5FAXGalFW1D3/HEmuCMkcMFsDOINqPpz+UaEQi3zHgUfKcCJo3+s3wnMCZ+IklcsYXRDP2BaQbHxQQrWC0v/1mSB+8VEWVtXPDmE4QfrxCP237v9Aemd76k96XT5IlmfrtuAHFaxY6kkO7crknKKaXHCglLGQ7kU3q4iA6lU0g5ecgkl6QpFqkqVcAiQDrGWdpCemNLoHcQ2inGaMSOtNBUqaAMFqPRrGXGMskj8sou79ELcn4WqsCDlnKPLSOUwVx+G4A4ABcXrD6b1gqiJ+kwHR4VXcQnuAxONBPFjkegLU0qXjYAjyhW63coZItfbmKWsTlycN+CUpkrZX8YYclWRzVO69uWnhNPdiolz78p/Z8TNHUqARA/div+Y4CKIn7fWyoevopbihjRjyp6174qK1Xd3vtspnQVRCAVb9vjuSyAgMySdzwNLBVVISMxg7hTxnFi5hjSIsZMfRuWC2dMbJmibDKGPQ3NdyVIckwgc32yHt9JXivc8FLgOwoGuxGFV8MK33iVSJeeqbLzVVUGb7yEPR5GpEP2ho5CgNMfVBJPgMKxEnytsGifAT2xLRbRN7IISgKRDLyHDZCkJzDX6EvFaKbc4Cs0r9oNT9pqYlE91hMZOlwF91LixnLRIolbjVA3vKMOZPwx36u4mNfMzJ7+bT1elvW8vnqxI+wBH+ECVKPwod9MLHjblMQ3uxrGWtC7nhhwvYXQwBi+wvENRk4xvQ7QgJZSLRPHg9q4BsHeBNwsI+wTwJwXiYyw5CHhJEPlqjzfJkO1O869+jRiYDXYfgO5jzmGeRc4bOwcNGAMPOXDiYatWCYMUsOO6qveqqvjKqF3zbAd0nIPLcM1JqpxMc8DDLXrlqYa1BCCzdSFRp9R0GxWT+C0AeVT6EmNQ+7x2rMd1RRbItNrlihWsd8V9IyiyqkHpWCBhaTMuPVIQonU7ylqmDI2bztg4HTOEV5ea8VO1w5ZsjJF26So6EV14+iJ+hwcBKRp8jKW5721kS/7pEm+TRJ0zseiYZCgkd+1SOPqyPfYYh3p5AXL/XlnZ5wOhUivu82I19UixqAuUJe3tGZ0HB4Xfw8eBxMz/33/DoQrBT0Sd3wa0zELNFDoqwEIG9A0qciWjxNghziWoZC1CDuqT4cwLv6pYIgQu6DN+qHA/4G9hQuwB6IukeOIjDH5gVsCrcMWhz+5wDx4Rdv96UqLHtKrzxKDkBgj4b1mK4+fDaqpWZZKqWc5qswu5IyTPOrdI2svAC78mIF5RYUkvE66B6wM4AN2IM23HlnJfVE+6fsXCAMm0mRi96j33npOecSLXVSZwyW82SnD5PoXHwusqKuipzN3YHlPOB2j9RHNsxhO8Km3Lm8TKRpk3luwgF8xLVkCYxjxpN86BtxkCF10dpxvzEIX/s0nemfIvPexnW04dD0szPHDRSi6ZR2SbBOuOV0WF498t2m02zbN1k8HFICHhWj5cwU8yO28EBVlpV4vIpaIevNrp5v/DsQmfMmP2p4TvzWS17nvMqYBc/mVBo7WJoa6JNMsKEZ5tOVD2kxOPFgZ2qyoEL9s82TVbTpMT9dVbWQ9UoXEUtlbj13tEWW8dYG8byiN7Y6EbRpTLwMktshwQ2qFHb9rc5bs4BNexPO8wFF5yZaNtWtVdSB79e8sK9WOUIVT1mIl1AQpL0WtfhklU7mgP2W78+LKn/s1d/qrYcvluh1Bt/u1hPfAdtWKX774TzwqrrzyIaro1FsIF/sti0iE56QZTxQv5Y45zXkYoPFBGnmlX2GDl7IuiNqUeeuOe10W87VIiZLk4Obd1pxzJvRIgpjWD+C8KIjov283TJD3pNCD/hBWvVx+ZZf3EXBNYIg/NIR+NjUvRTplkY7fob7sIBSYSr6fOkBxPcOlrSUSFKB+fnQKOLX73j0jNdzc/Cb9zsYqodnvBZbSw/HenOri52BI4Yf7PAD7bz+VvB88AC6FHJWMcppDsJPfer+IeYKn3pk/hsJNMF6Rxd2kVV2d1AG/o8syZco8MPAq+Bp3PQqKrI5Jonn5rajRAByVnjxLInB7wVRorZWk5CeKY7WA4BWwNDP23bENgZL0/NhgLaFYe1Xp0BgCeV4n3b0wL8zCNJsC294ZJ334pG572VIgR9+2VFAuDdYCrvzyaBGEH7m0xg/3hz71CP1H2N8NR1d625f9RSGH7El68JuD166uep7rqCn2btWF3s6iQ/1nqlV2F+QydX+9Z9DKv816NgwWMtKKEeVdue0cgPb3rv7Hj640jAY2fNC1hpgmpbrjWGNTzoiKjY3z7jTjv5Pr759umtGByMSdDbl+rYF/XNZzUGobIC5Ls3K2ewtv9Zp3F8o4g71FLT71PSqG4JQTXbaF+aCFWvYwkdjSjzFOFIt9C63q6ZQFycFII4EQzoIYiJaxXVXkXx5WZl22jFVHto7J49FDZnnqZjWZLyAhagmMhvFnDytST33yDDVH2T3hmQGH/1mX0NBqCf13A+BbtynF0g8RYwOyL5V4574JA0855yXwoOWXvGDG3YroZrwAygd5VPhyMiMNcdt9mUUilMQIJh1cL2vRMjypXoTPdezup+2ukxNNHYJArzpEoswqVaii3Guw28ELn6zI9WUR2icdIUO+uvDrtD7gq50tCiQ1G2lsKF6HlgX4UJTq1Hioq2rWiM615WuBpYnXeu2QiYD0HI86AjM52tazCxv6eBT39s7xv+D8D/8Cr4JHHRWmsC14p1e31IoTgG8gf7FdRo2egqGFSq0RVRs0rl/Xe7nnKWDSo/lBNd8CjCIb/gEqmTskpaRnxzDytSkYxyINQwVc7KubroiDH90pyxHCPNWBy8fxiZXR0KwB902tBJ2NKOFiHrI+tAYhC8xd8ZfTTAkCsLL+wsGCZR7e8fqryC8aowOnJg0r9AZxVfDZFI3Vyndj1Ap60LPvh9h/f6bcK0L8w/ql9xGJHJzmhQXgPpaFRqyDML7jlZe5Hg0VddjHX4QesH/QRMSQLzEAVxSxDoIbw2IIUExCF1WAipzr6205/NKXheUd/439VvdTB+Ps5Uv+eRcxXP0aBUHYLihSQYoqA4lC/RXED7y8g/UQOFqcw688KohREqZJhElUNoH+PvNEJHJUnw2Sn9QVE6fH48qQ2qIrX3Pq51MWzKEJ14NHsdJ73Y0FQXSP/BqJtWGHHgn9K1OMRXSEC0cO7+D8LxNtjAT9WaffEHzKrQsDFcsJawMRnjeq6p1khC6M8+gbkpeVs3U2TQBk+sk7KTScs4/SoB7m5fnk7c7h7KddtRvYyFdgLIulaiMbBfXkrdbDG3qCYyKHNz+k6Kpti/q3CCDQf7AwqYADnowBhgjZCaE7Xny5M0zv05OJV2dlJAlyohOg0esqmx6fjSuTFkrjwb65PIdLsBVXnIBRllfwBl8URnZ7BfwR1tEKM9jrAJ8s4R7cZFKjwG6mxPa3x+oBWwSd2Zp25fMLK3im0ceog8Pz0ePD+Sthw/kbRBumTZj8HaVHYHNIAIjNBW4+HUOcSVBeHum5uIOZon63m8aMYHr6K54VSEVBzKI7O2pP+50GzqYNSPydIk8WSKP/HItJXSu9b4coHfe3Oi1t3G7Kz0Z5Dn1e6Lnc63XrqD0r/YEsCaqflqaKeQDL+vqgmFjAeOo2XpFDZg7R3wtW0N0K8d+QRCOkrkk2XZUpMacWbN1kRO7qLIbdiP4ZS1OxQcDlC2ADwNkLfCx7A7o6JOiv3eO0SbtBAzCcOAgovNZiGhX77UeDmgqdJ5YTJNcSMUfk4gDtSu8bAhlFDUMpPGYNvQ+FlPAMzkEyTUjmQlQftsk0X4roEw89P4CvSYsGWtdASwmdBU12cugeslpwP8u23w1AKdWivi83VbwhAhkzAWC5Q9xKHmW2gWtYueIhQbSlyCF44wipcHtMPUIF0tGF11ZIpnCDI9v6AbYxCWzpmikgce+6PLU5OLgnGqpYnmQyPmF9mcyy4okvmzR2EDkDVJZr1ttGiCMntelvoTueCayhSpqP/wU/+i37/basdad9L/tt+/22l8O6L8c0N8d0N8d1H+h9V902333tTtwX9sD590a6GfL6uda204E9/qCegJ9xPVWoCrm9SF9iT7mopK0PEj0WXRmkVuqkb4CtLoQ94qhU+SaIyWaGBwbtzxcSO23QNw98VFuPi8sO7nRa8bzIQGgS6QUFdmkT6QE1El2a0uktOWFuutcHhAvwfYLHUT3/VRJCZ8RjtJ0Z/ucUkm3k4xtdXiA9EQtEP1Eb2ef+XTICcrEApnPjvJoXhXgYHI5eBQPUit+7BMjG5mi61FFDJsDegiLs0wJcf200oZHSe0AfJxK7vSBqIC3PGp4CvRxjxEzJSohJ2qq3/a5kdQjbB1iSzSuDSmEfYFFo6S9ljCDe65CaY5chavhvuDe+99yxBGOJkTxUL2gA4PcK8ksh03bmueImXsWJHnCyg1+iFinD/tiw/MKZ8mRr+juMq1bXYWp8v0gg/t1D50U6ZyYiRqA1whoU+5LhNSkteY0kesB8J28gfwVFc4hXb0rs0k+1Y7lqknJ33wJW5UfNxX5rJ4Hys0QuLDFOhOs/eM8Hssj7fpZm8qWvQ4sQjjBL2B2oq3Qh1NIkj+tfdtJJj6cZkWc3kObyF9MQObpGZXej0k1F9UPqqODaeiyyABTmfCjFVz0JZc5C66iS6aV4/+d1ljg/2tuK4HJdRt5Hon0qsggInB4BPsZYP+KsQDjhsWlFctaV6ohrANRZylyLaWDIQFFg6VTegg+OwgfjJVbKEP9ZFyw+kcp6vOxSs1Lk0nFq6OriN6DgCB7e+3fp1VZDdT1nDdpNOURXGiHHAo9hh9GR1EqTsE8BKMivYQzUlsSWTT1Q2uTRo5g8oCjA0tntMvZoBbEVTz8u63+cvYaS1GjW1rUSsyEWdBv61GA2RzvZ8MjVIGJf/KIYK5rqZbYvqHzGVP6dlxHB3dX0ArC74a0EE2SSfh7rDNbLQi/GVfL6nJpV1ldLr0qwWep2FnaFakt7U1n/C3pjdSC8J9X620VrfseJYp8Qk0ghgt9g22mB9uDASGdQ2U5D+ngigxkApXmKOsWY2iuMV0wgMtDy/ZKYQavko6o1lUym4lK3rGV1OKqeBHhq5IbrpxiQhT/0g+gpaJV6wAceG9IaLo2jG598rjjIZHFsjbIS1fP6u1hpSGJppsipj4VYdbwP8fDwiAMLOEAw13epOlKihSdGFOkEA2GIYLw0TJF8uPfH1FT1Eo3LRWLKI9iR7f9QhhOgIarGdd6+LK3OwLnJxA9OXx2RWwgMq/0JTyO77utkH7ZbVl3GmgxhgVPd0ihZs0OftXTyuN4ZyXePNffd8kBXsZEAsO1pgkbMMPqdqe1w9l3vSPGHDU80GFMM/zJcVJBYgRGEn5pa6goHo2UDmGaI2s59WwZFQRrTro7Xg0gzyP5c4/cVMNiwAW+fDX1BOHeuPqY9NmSQ+F7pg9TBuEFMnsNVn74YvxoVS9I3ic0qB95DjCfDhpseGvBmJouDsCHvVyRzv54uSJGLe/79CDRlef65XgpB9PUohx02foU1molyCJD5NseD2CLN6C5ApFTc7OriCn+u67SJ8PkgDQLFpX1oQWhM67V/KWnlONBWTunaRl8Z1RiOM22bw8J6YqdEd9lH+Rx7JBB4iZ+bw9Zj3gtrntkQFlVVz4JPJk8di4H3Zt5w1uuWP0OkQOonTJs/lGlwecFwC0C7E9rstqNetntJDcAl9DxkCgIHwwfZaZ7ItnTxr8GpznyNx9e9zAqHvAqa8obHgm+mqg+3PDIMGiws31/QBQ3WXZEWSG3B1R4VYFja9Mj7jVZ162ZDvFHEK5bDI/Np2QpgafhB6e9sxMyomU7IUeRmBBNhiCciZ6dgQl67GoobBUFuFBqcHCYupezRMaFsSHH1Ykh8lj/CTP9cnWLHg3Qe3NZFlJsxR+PUkx2tTe92ngmrAjH+OTDLrMkKahyrGlGO/bNZVrgOL1pKdGMYWoAs+y6zUJpkWEG4WX8IfJIbbFhWHfbwK9xDhcyyFrZ2t3a/vSk/vlP70WJWIsq61Bt/OW9OiqxDANnKeCdE9VPQgCZFzkSNv7w9U9ykeJtkyx4igTdiHba6fUP79drBMmRPcrL79+P8BNBWL2kl+/JIKpfiL/Tf/8HOx253vcbkbwuuiPyu/frSAJZeZ/c9E/v15tKfk4W3dt8v/f8px++Yl/+0xc/AClpp8PbyKs4SL+2R+JVMPrqAt+U1G7DX5ATE1bLtRZqpnUTruuSzJJX+4qaA0I7V1U7+j+2FW9LUV21iqSxlnRBmHndZmy93muFsgiQ7Lacn4Akpd4aEm4xKI5GP5/oMF19spTVk2QQzMbEnlF9yB3B3J6XW9tNupzmEy/h5GRKnC9nwfZ6tavghk7Qr1/ZPJvIKa72Osi5KROJmbv95iD8de9IXs0aBN/uHdyXBOH2u/KIBuGGQwrqJDvvjJCM0p4KTAakGTUco49HjlFYNih8tlQPk/WUhp0Q+ejZXd+xxqVktJAv/umorntN4Sg/qq15kYmMqBpFzo4SkcZ37BYFqp7xJJ0UhwQZgh9xyayEAYgknWJskTHgOT2NfxF36WkGeSkUTGmJU+V0+5pm2ZtwHTiS8yQLwtPIiBXNq7xJTwPaVsbKpNieXmj/purnX7QNfeZUoke9gsxNC+QPamHhzzOIl0P9LBj4LD3JWIZ1rmdVBhxaVNVHDLyV0+0zTEErwj1cZGwKqGeFuj8RQwvSyjktZVMfRKblNKPIByQggnQmbOlJxuJkUU93bnfZWb/GsFH8fYEZjo4YPbHzJGu5Uj9zmVFLOgi2USrb1WJVBbdLhHnkuZC1iH+9ChXrnANgLM8mKjWlbCZpEqHyEz8ra0Yv1uFljdi///sft3/lxZDKC/Jp/coUf339/bdbCnKAcuS//uO3O5vsRZT+vv7666/49//KDpOtXdKAIkC43slPdo7p8DkeAp8scLHCakSMssfdpiDc/Oabv/zuOwGAvd98+9fO3Kum+cVLABfjCGgNTeRRIHyapspb94GCPlQwXi/e7SgglF3tAOUy1YS4CCh4Vv2gN3pa/RKHSR0cdApP1FJgAnKaaedRV1HRXbWplRj/HVKzY3Bc7j/uqikSG8wY1wHpopR3B/SI2BPZfAcUgNDk1oCwEhXP928MSGFRvjcgo9A3vD/9SElw15dBQqkDQBxXxdcUTx5l6CZoZ8zAmay4QNtyybeUyYKI6+6A20VV9Vo70EoYHMUd2utWATn1MGHacJXSNIvhY7Jm6qZMcdP+xzEmR8XsNErnaOsE4ffL2CWX9le4Hf5utMPpkr6mK11Xssp1Je9wXckSHs2kWK2bJbeXmNu7P0KgSMU02w/HVLbZvKiSH4t8Y1jrshZFZa2j7B80eVys1VIX1NBgxtREZEWipUTR3AF1Yea0gBKyPjC04JjI47NtU1rMNttfwA+PVnnaADgoJjjQNLfWKhG4ZdpkudWPKKYX2188XsBMf55awF4/KKpYnqDf52rJrIyFS87PrVcsFb2mCVJIRs6NPUMGSpkgQARGL9E2g8w2KnoAoyJHWjSe76yqHAtDSnJNNhC2AHBJRPgxaWEXZd1MMC1Xt5xW7wbeyj2VHqIgKlnGcz7TPGtsFgG7GdIUagxdTG5esO11T/N8wbZ96nO2veFpRuM/iS6QKImjmvZ5qot/+3LyJRo+P3zzW9B/PEZRiMDbFBTYXKo3E/nDUSU9Cz8Y01JRwEdjOjNTjTnalYLHPilzKubWHFmGUlG/5ZY+q+VTNMItD8VgDSSBPu7BupYtlVfvCFU2N6KRSEZUfC89Grjq+s+LoiB8MHyUOfc3AzoEiQ/L8/ApWp0gfLZCP+akwZAy8upait47AAopS2fTo0NGh6X0mVeJ9i/+G9TSljHNf6w5xe89ah3iOP+ZOkpB+PEqPZnz+ng4M1FXSTQwOpQwCB+OHTn66GiXV3H/CbTU/+jaY80pfMNb51sMifyDQx1let7x6WSTgUENEv9QpmN6LHlB+GKpcofD9P6yA/xPvCUr9V24lvqfeHvs6KBRiLYDNKok9A8ac6Tp/7FPy6F4pRSd9yGRhV3Su5DItkdyWQXh7rsf+C4H8RycNWUSATXwigcNPFWL2Zae1pslWv5nxbS9+sv3OjwI3/20lGDw3qclmuBX73wcvt/tdz1shMq4c4h5t+beVj2ZNSp8i5/nEL8i+Wo70+eNvqLhze0yAWd8XwwzAWutKXg34SaJOvhjrzhvsl59JmiHg9ptASNpPhjUhHJG0uneQKujiz9B65lfCz12UNdt/WztLb8ydHfHq0EZXcNy8D2T3Ps89XW0L6XDm2yIlsGAbg2bYS1e1dpU1VmMlFaM26IhbmMzbi673Mbo3dfPppMc6rHJBjmM59vF5qCOwh8C50kwqETQixlmkzSVeDSoaGIp8kBUw+TKABROX0fKj/jweVu9v4NLOFxBMd4qsjwZZjs2mn5SYJWoqgz2x14dQ2gQpQltmQZ4kcGCbimSnviV+o0tD7mHZxmt7bvDcrK0/TeHCsZn6edpRh392Pz3ZaicaVvrP5dRQvKMwY4UvDsNZT8rtAIowhQTei3jijLJZ6nAd3xvQFEo30UQ+p8kEqqTyiOvQm5qFGKizLrhVYOY09DbVLNALUr/sXT6zvV1x4vFLA0rPIZSpZ58NMU9gd4zzBo8Ol01kAqLgb01eEmInGuhHVyt0jYtPeUHyod7vtN8qUr1HKeZiUFFu2TTYlbeqQBoswbugybHCRMTzNV7CsINIxeHQO9AYnTrd0XWdHfOcadsrVUiEvAu8qJOpgmB+QSjzNCW2+ThcsWZyDdHtZRL5PGoUusTGe9MOUUuOdTS9dtt9qLbtMVeXOySWH/ba9nqtXSPetU76qWnpdvPy14/u72jdntH7faO2ukdte1p6R611X0c+WJa7rpNWI7cp+4+BIfs9LKnvd+Wp9e8FN+VmF7tCbzN29R8d4gQG/wwv0tSMcisTQyOX2EqebIQmyOs2N9XScaroyBcA85ryFzFGjnl/7wEjbgbd5sIWE83XVZ02QyMQkZZKxewLW0Lc8MlbNjt57WMN1t9X0t4s41n8uG4nvoQg3Gt9ktc0p36FG+o1GVxKKKmLiobduKShk+quC6rvmntmihpgcr6MhEn/Ie8yD/+aUNN/7ysy9WjMKOMxEsIvLH531fk34boiHLHq6Tl3sURdKJJc5P/9l/W9UoczuIQllBy95i/g/DlOx57gPvoVal/XTbiVal/e3zRHurff3rPvjDQqmowUrEQ6ZrDEVsXyHJ03mlM4svOb6on3uySwlKMjQACMW2uyfdvd5Xc309HxYxsKUOM2NEVhyWgnaX8xyPFCCqDsHdVCA8BWQdgy+6zMhGRuO5Tws/7EUp0aILTO++33ery0ToBjT7HrCN+OkTqCijizSK1qS8HdQ3bbNv6fCVdADRDnRdD6hlg+czSJOvQcH46eECSJ1GZLYa4aHfe5UDM3tEMoaPHoCYxse8t0WYA+7EFn0bvRIOPAdZPBACMZ8J+DC9XO4CgB6qZKlPYfScm36XX5h6EkHSzKokHh4B1gPXYBkcXZObDgLFu4NUy3bbR8A3zbOS+1WGIfdXq47mCkYO27TON9b49NByfDR1EM417L8PKgDMVyZ2VvlbFO2PrhoO6afZysdJ3PUUqPXhgtvrglwMQOKRsvdbB+2u12+ZPBpUV9QiCwq5w5YYGZfFO3e9uLVZ6N1ExF5XYXqw0e7XK/uEydFVDI37o+fuHukPIrdIx6FhcVy4ZIm1g08WCmR3T1AOtHZAE4QWbYxucEzddim3DoYjL4w1XiNVhCxUMu2ATcIPktkXl7Fns7o9TZ8PStEQFTtIn6XZU0M4f5saGk9welOLBXw4RUh/ME1mO0lorjSB8urSPtvW7IV3jWpDl/shJbTW9TiztrRV9MXoAX3ZiHoRPlvXQnZB8JON682Zrf72cknzw4rRKED5b3kt3mhpVhkGoU1l2h9TRcKLCD5Ta5/j94EFJVmj1wRtrlYJw8GrtnlrB4Niw1e3b+/XQAR52+r48CMMlx3d43X2amqZNlkJE85FTuopBuL1qj10TYoVD7Gc0OLwosblxbnNwbjGljcP3Rxojc4vpY4Xvzejat/LNkHZc1HInL2oxcn1GRxvs4/207YPTBwV/+laBRxV4wIsq0wXM+oauWfqaSxvX2w+AKPJcxnEpjSUyqFs/Ia1wrf2JOTWI73XdbURsVoqSddQBuqSogNm8JRo37nRiW2cMFmRePfIrIdIebLGV2gO/GrrJtM5Dv462TJTWfb+WfU2v/SqUM/62SSqht7w2s7o6+MXwwZrWgdcij470AZ8MH0An6eg/HdZHxngwprTu58O64hBo1UWsr8pzKzAmsnLHHiLqpx4i6qc7RKzGzhCxJDRErunG1pPL8FzngT0k5ZE+t/0bMems3+3ZNzqt1um7B9D54XLjCocuK1TuJpN1nBQdCRQbKQCXjgSuzi+hki2UbJAEL6DT3XpHhP0Vs1n3EOs83UPwRJ5DrAu4g9wqURH7vsQDvhiVZ+XO3TE5nHDTp0Bg5cS7AmdZrpSVOw+XKo2fDxGlD5acTyuNnk8rwfnuj2vB2ZapZOXOgyUqcKZbgzpwkhFpVu7cHpZC1+b0ojLeRDMcVVHAkA6OvyU6OOCUzoal0xnyXVF3aFsi7HExld1DfEMbRPjFE42V7nlMDt08HJG3U8uTJVrWVLOsQ5p61ontfl40QL8N9DGUYPMEyuWMR9ZiSNIp/m3SxgNXtZllfZ2nIzoAcY62ApgT0+TwcU8X+WI6bUj/ZbfJ0uoTVmQMoYeuEmBx93qDRuQGtRuJOzHOedvp8RKNIAyX9qGFnZuclKLX/6QUyJXnli3Kg6QG89eE8HqFjS1FlerSLwzCh2NHtiHCMa0IebAaKV72tNRq77sGJULyuaGj2uKJYR1z9ltdnUpaPXil5tjPu1LNTc8IcXJMHISPx48217BEz1zNZlePEmLcRzKmZHoKekqFZJMGtvjIhzD8gltFS+u2R2up2FzMTlcMBK1JLD3PV0mC8P7gMeaswyrmzL0LyyQfu24UDw+OTGSaJXbfc/G22DM43KOHB4erZ67myxE94iYduyDQCMInS/swl7Vc1VxZbxJqVUcvyjMJ2UcOj1Fby1zFx30t7yQIzUF4x69tzvmoK6elEkpu5cpqw1NUysemqNR+ur1hvk9UvmMPSJV0u7PQqJY5XW/NgeINZq85gxqmj97DpQIQ08OAfPhVGmrbfrPnVWpy26Gzkdyc7ROffGvgdFtI2+fVNyccUjBn7L2JqCoAdch9Sr2JHpHAoKD7/2PvLaDcNrrAXzHDhjnZ8G5wbS+mSZo05aaUMjpaW7vrrCmGzaaQMrcpMzOmzMzM3KbMDClj3jeSbEvyzMj9vv95//PO69EJeO5v6M6dOyNpNOOCpldBSZhfjSXz1tHPEBadn7VJAlinXCOEHn082y3gTNICrZf/uE5mUb19Pm+Ox9BevYQhnKhbDPHq3thor+7l0L3b4VzIBnAEX16nuAid+EuLx9BDkINZOx9hS1MiIENQVRroIagKRQ9BFRRbKMgQ5I4ZYKgOVS5FCE4hSwCZ4ZdioL1tiUB7E/sAJuBSAjtuLGl2uaBZFaj8xhG+z5XVlsMADjlQ1jk3FJwF2lk6e9faCQMSbh+6UbrvGebdkaK/ELOfrw6tDgcba0Nw67ugkd7wwsqsdR4jiDIKLrKiIWRW5Yb7ZGBNqLWB4QiYwEpuPExSPiEDJAqNa0nGVkvAEoBusB2xkTLHocVW1hg5KuNeZKF7Awvda0tG+yRmKuucFpxJm5MQQk/aqBSgTQCEVtL1EIEnWVhMSzDGK8iaafAtkpm3rW40SgraHRnVKlIjSup+u1ByLXjSKsqkIAr0o8Ck8HWGGZ5Lan324DOsVCKXAY8Eon1hu5xj0XKI3brFMLt1y2H2UZbbh0shhCBjVEQrV5QQZu6pRBrRv20JrqvYBKwRrGMGop0Jw7atsUiplfE0pLjKuqYEopbqJgdiwL6CE4OZkAuDdUZnAMhGs6NgAscDjIbIyrYJS9HS9QiIoDtnmnF/yyTS1odsrkPsfRLQKiOhEtiA44is6vpEdvZ2rAlQkUeX8NiWaJRfZH/RaLVnVTRbBuqHEIEKIlKEjY0lmVWS8QiZ/aEUGOd9LWSmsoWVMG3bAphTtyWQU829gJ2nb14Qz62ETiNAOCgEJBioA5KKVTJIOMywrXCrOH5BIu94LH9KQAByhkWADaKWoKrP12MgK9txGABoFpcAtKYAgDk8W2DpwCeJ9RiJtF2a4TAJaBaoAOIebUFl1zJoVpb6JsMkVfobj6OsItfjCKBBbBKWpiZgCbTKYF7BkViRfLqxvtqwvq20Sz4GKQYaR0shandJLeWixVaZx6HFVsE9ndCaylvRnC6SzYHzbDPJRGylfS+h2eFpZ6NGtfTT+lXeGNcsPXLwBYyqApy7kVQ2ObkkA/sh9veZObBlj32WWC7WU9pofhKMKh1XEeux76Jmu6EVRl/l/U7cjLkTtU8dzNfMO4WYhef9xZnpxvvAzVvpy+QwpDA10k5RWnC08xisvE9Luav9o1jlO94ZuFj+Snuas2AkexMmpLIBFKzRS5Q/wwYP1GIv9M1B8gwGnWynIkB/zuO9HCRLDOHkVe8nsPq0vmaBZRRAOZk1wCj7+YN1TFv5v5NgIFbxYPeImvTpAv0JNkI5WF+pgYRXuUL6M/cXsjNpJlOQrAM5J+MpcA6fLVg/G47CGjiQg2Zb5vzZerFUJplMRSC5BmFOppOhmD/P6V7KSCYLPYl0L7Sv1sQ6uU9Ds/gimOm8mc6Ab4NCgcl6WHwjmmCFHXgYGNCIEA7aiGXOn62nl4KFFW2QPCdCIGzHA0wreGMO63gNKDKwbK21lK21OiEvk47DmmpCNYN1ozmzy4gVggacEoUtEah+UM38DKzUNoM1rNIqBkhSrQjO/yoza20pVJ4O/LNo5fnANHg02EAyBY5iDdD6iqIvmcrApgQNKBLr4RwwyMNBMJiHK2PYEdGi0tYz/arkpiNAmAqrbMBhA20lAtdgdY2h+quyYIhHr65viy9PmNuAgkjF2GBtimmB1gSSeQ2znhIYaFfh2gwmDDOYqiqE4foL5FBtG4ZqpKqisJlTlQFAvLxHta5PmvAGAAVhc0UX6M/Z4zyz0JkaDnFyG1+FYFWVTRqpDCQnPATTpwNhRymLgc3KAijYWFaigjOEjfoBFDLDoNvDbE8Cbu8BFFSjPQmYrrAMIh2YCrAMbEC3GX/1PUNn1kh3F+0dzSA51oI6GTciUazDyRrxeNK0vlvGmn0mZz0PwJt9FQRTrgNhfWoma6bht15BGMz1ljHslANQRiIKN8YaSNjkxE3i1euf6kDVi5gPTYRA2KEcMEH27Wdg9m0zgRWD9SU8hKoYpDd5HpmlMylwJo6ZKyDasUYa1qcqdJB9wEnY4JY2M6gUAznYcF/h/Gpq8mKpTCGXSUd7osiZ1QxcBGxHqsCBioKSsI7kJrH+rAQGa7Sag2u0xOGzNdKdZqI7cN4G46DZljmsqitfXAepGk7CVO0msc9OvPvk4B/foFjYcxYvi530WWdJ4yd9PgQ26bMQ7CgN/gaHNMG7ai0ozKN4UKxfS4FDa4ykffRac5BfQ9NOKWbgaKydOzBslhXIOZlH4Jz9mMFacQuWsLt/NjTCqlcVx/2VAoy2auP8MwWO4Dtb+VjywM4GJWFTZDfpLl8DCqqthIHPy+Ek1B24yJoybwrVmnkT7DkeVD2ADFRPE+SR3xQ/GMvCpotTIVgNNl7NwXx5hcPOlMB+FIEzpSoIXgELgo3qEyEc9vYsZSaTRZjKZsGocj5zC7miOd8VMBuPdxnJvIefBOP9JR3kgYxsNpHunuQLSwZOq6sg2OzTgfAtaKQ6jeDcfBDsjsWCAu/CqynocGrYu4fDX4jWgkLHLjeKfSQEdp2o6a2gC8S+ZU06p+7aqSWT1se2U6BEVRg0IXvxmm0+eWhC4PtdB7APLMs32kfJzp8/ZUZ9dItto5tusXiTaTM8b3IdrY3zJ2hvEm19+Q3WPnjUav8fceM+qYoMsloHwvZw6/+wOUUABfPXJQprEMmkETcQ92fBIMxyXGCQzuKBHbQKQig2HqjYRMra8y5AsVUUVLEOhb2tBxsMBd3W+xnYbb3NYIdSsL8gvAWDMJi7K2NB1Qt8KudnnNyaqhl7pKmsdPH+9usjaJhMGt3FNMy0AihoWzsUdiLem0kmOnMJ2FwrkINNUioc9kbIOoPS3iYmaDBBoLD296DueV4jksJap31SZuBiBgiGLF1v0Ny7cvRm0PQXTsLm3m4Seyu+LJE27AdZfbBa18TC3r65WFjXq4mFmfiyWmZKVRDMETsQ1nEsMxKw91FYBuYXbQY7wgAkcBJfBcErZkHYkdTehCUJdXlTESC2+D0ro32wqQAeghXfgQJzg5ngVAgUeDPjcNg+2lNMrywa6Sh0FUANJGx6XSLho2EtKGx67UGxA0FPLhUt+FYYwAYCGAcbCCocNtvuXKbPrOFFNYyDZVvhsF25O5eBPerEMrCubDP+rDq8iLP/obMBzCq00PeAyh8T/7QOTcOGBD/tHi1n4EDssF6BYV65FhRmxB60pvzhhgJVV2kzitqUW03DHoX6aewko7yndYBhZwtpM9MfNBeBYLC5SBnD95JsIfDdtM3gawhWeedhFhGEQa2hhMFc+Xg06fwzGUpgh0ewk3gUsfqqDQUGLJf7h/HK6+WmI+IFLmpysXiHnEw1IwahQA42rlW4wDV9FRRvlckUbG6BZaCWazH+rDzPI61tVirnxAX6FD+O/ajER8OmezXzsIep1Tx2FmXhQXO2Kgg2Z3Og4Mo3R418PpEvGGnYtKNmHmp2VbzbCVTryYNib9FtPOgWvZqC3b+UKLz7BFAEdjsYhEE9fwkLrmTgW6cSFZxU4DPoagqZYZAhdyUTUHvCQzBDdiDs4NBlJGNgPQHsCUowCLtTcYFYxdpckGKrKZhiSxT2tsfsNzLpqskV7LYHTsLmoW4SW1sbDKptNQWrbYnCdjsbgnX0IAzW7coYdvA1i7lMLet4YBxs0ljhsNNmM5dOmM3RlhpuE0toYAkhHLSEZQ7f+kZ30gxcv1pNwfpXibJy8b4xnATjsM4mDo7ACPIjDoT1I67t3fF+BArC6ukC8VXIFPIw74WHoPW0IWxLxruSRh52MxtAwQb5EuU8c7f2hqv8fwYeDXl+TEXCGAspcdgeUzp0J6jHwDhYj6lweHsqnfUT9LzVBQbeRLhY7EhROWgoaKSAkzCn7SaRazjcUE0ljAQucYGTsNt+NwlTJbQ6ZRhXnQhkzgNtxLC1Aru2BvezsOG5zAZ6JRgIexXgAms2N0jtvX7HjCWC7gj9DOyO0GbwWXXmYM9gsAw0K4vB9uBYJpUCBzMGfn8OBWEjggvEdg7nKK8aJnpwEtY53CSs4RtQMHaS5oBBkzQIBpuklTHsCBbLdPdBvykIoGBOrUQFGELczPeYSdjbhmAQbghlEGvtsWQiG2TtfgZm7TaD16q1neEqf5C117ifwnsYm0HuW1ED697i08/izbHHKHQHf28CwaDmWMLw1tFjpMykCb0TDAah1lEBsRPHzmQmA6srHoJNHB0IayCdiULahN0ZBVCwbleisMaPuBHDMtDK+e4Q4OlA7yImVDP+Io/2IEZ5wxtrp85GhNApQrTHiBspIxdvaEQlA3Zl8lqHS+iUspzMWDznK00CHAZQwxgDJ2GzFTeJveGtgLB7ylpQ2ONfDwob4ybjadhiFQ9VY51gw2EtKL6ZgvY56TQSsZ6iEfglTIXDdncDtQnFJBgVkFSskIDt2RRAwTxHicI6RiMXMwOXC1dBMN/hQNhBx8iauUIRtjIlCIMNOmUMX8OuGr4zrIKgNeyCdtfhLgYM3klw0Hpnsau0daAjiOVMcKiJfZSPsw1ZKmVky2t2U0bWE1zZK2yhE2yfCpUzY8VczkwXKrsnr8ID1g7d+BTSFjYjCAOrgu2t0KxNyvGwtbNaZVbk48AhrUAh1lFkiLRKjHMIUsGYEMyNRSLm8mguhS6Nudw6liSVSAczRn89juk1zew4HBBP9GHLGcNHN+LxaSi5E9KZM43eeGZFGtMA1irwYjZuFEw801VMJq3TUjCMZRVjUIh17tMmHqnrmA+UJUOQhsaNglMJIlpQAKYADY1hdCyUZDI6ikv5QZSt/kkBlNUAU6FQvN/f0xcEcgFAQ2ACTveHFzve7+n/EwMg0LHHoRmrZ4/HykGXrMcS8UQfPotYQAJGPD4dCVR3S7RiXKYRANmWgdZexTDGIhmra8Jq7lY+rOaegmIAu5DwHCoFdPYhBee3lk/ITGa6dzRNZ41ab5+9iyY4aaPKYcCEZY8Fj2l3hFYcgkkbF809SgaCQMfFfLkR4aDTlxqrGagGytr2kq4ONxYtB809Eiq2uhk85crgiZcb/aNRctBDR6GE8UQfskwxdDQjHp8Ck1X3RihWHQatnasboOV2L4DqvdIJIEZgWwl4zmMdyNTQOALGWJ14A58EOrRCpQ2N0yFSeBzrbCIIW+0MYIDHGcAASw+ToHKfIqb6IPB2LJozViDGOgwXAMwMTMDd42ujS93e7x+qae802gv6atqOZTBCv2PxRnSX198yXtIpq7/9nbOyKu7H3/5uAPgff4cuHbYFHNAYpAx4EL97qUjjiT50sjFMRCMeh5kazIvAWhOqeLiGMF3IBcC7kAuwDNHvLp3zyICfWFQlMgJ8hZ+AdSp/GgFABCFHZ97Q2ISMgxBUW6tR7a0CIM8cCwXZfR/J+DyXf0YTx81o4u4ZTXVHhSsMZh/4GUAcPwOIo2cA8YAZQDxgBhDHzQDimBlAHD0DiGNmAHH4DABabMzQHg8Y2uP4oT1ew9AeLw/t/sR9jt9vwtCByV8Bv9f26RDnsKsezIxEy4dWiSwrqs6tYkFomdE/HCYDljMMJogn+qD5x+C4EY9P8IdXW0pV8VxWApfZFlKlpIp1VOnXZxlD/HLLKsb7Q+1nk/ZTfCumM61LZOxmjPZk8hUX4wktTwHLoeB7/WoWhFpHykFY+//2A9aRaGKCTxQvplIrvbFHYZDGsswyP3+VyoENjWNgZDyRj1kvknzp+KtbDiz3HC9p/dcu7XCU3BcxbvYlYqarCLvY8tJZNoVcorsbnIdlFMDXi9Pq84X4nDl9ZqyQyc2toLNmzZo/Z040ah2EFzOtAz2i+WRmBYjYM3d+eQJWSjdr5PJgUwfr32iumCyZKRQY7pXZp+6B59BOo5f9hnNMt7ncOtUE7EkxAU0YfbYNoBHr6TtA0Pl0o/KJ9Zglu8+ZRjKZiVUlYiHW+S1mP3hzkB/tIzw/x/iEIGfrjKl8NFEoPWbzSMufL+Stj1fSgPMXwsclCqX3I5YTAKteska5JO4wp02s80zAWSx94OQN+zS2rlTEiHab6ahRjCcy0Wwia4JPxoHazCw4tzuVLUwIoLrN9EQMUjmzBslY576ACTCSAEqJJvLRgjNiIUGgpEyxgM+vK1nM9wBDRxJxE6wdmbUskV5mzJnTZySLZrSYjptdibQZjwKXURXW0FiPxgsrs2ZDYwMaAOUGDriQSIMbJTRoLi8m+oykmS6M9kCFIjiUppzVMK+w/J+h/nAbH+kPzmTNnFHI5ObNQ4rGzxvuF6UzaXN50UhW5ZJI92V6TUhw3swVqsraY+SjvebKsf5wr5JG+cVGPpov5BLp7obGERBZzszmGhonQiSZXNzMgSMDOpeZsUJDY1W9jDz4JhyiJnAaXjJjFOA5dmYySXgkI5czVvrb2S48IIrpxPKiGe0xgJmOQUB2y41HSCt2MgFBuNRVj0ZsrWEAu5Je07YV6eonpYCGxskI0Pt7EoLCKKYC2YpBJZHIW9HB6SINjWMRkGOBqMJ6DXE8gqo0wQgEYfgaxyWpNM40NOI33Ho0ajeTt8+A7lpR2ESozFvVcVCmUtF6qNxVmSkeoHeFket2mYnz21+WMuZp/tFwxq5MPVxYKam30RLpavMaCSPs1CfARDhNOYhLEV7z7CqmYy492D/9+ZQgTylHQRG7mHCZ7YvHQWUV/YyFyssOwduOlg90l9/+7W/HMoZpxzJjV2EsXFgaaCbBxbj+WYZcjTEOSdjdxpsNCHNV1v7pb6wShGmsEgLrfiWZtyb1UMZVEW8i1igToO8yY5cCIcxmslGjgEjd41Hr4UzFsIbDAcPXIysCVDO5CbuZEHlXxtsZDlBMGzlrKp8z8/Za2FX+oIrpVcN9iXyiMA4ptjU5GSk3+81YsWBvZ1fqR/b0LZkomDkjOWfOKs/vSrP4MK838gnd2aglJLMsn0mXRpSCmS94teALqYyxVailg7EoqXcIrhK7i9ZUhnLVrVIdWGllWASrVOMxgF2wqRjCXbbSrMaZ8FSaxxtQUZMf9HZtv9Sd1QQvsywDjhjN2GfTmikzXTDjw8qIUbCCfK3gCnenXLqLyWet5wge/VaFVToZBLdyq0fL7dqWOlk+mYiZvux8QZVOVg17Olm12NvJquWwTpZ33kU5Slrl+V3xzj7MKshouMzbAX1CaCPYS/y8WvGHuRqhGvc2QrXcLtIUNADrfDmz0JPLrEhHzX5w0qldrurASuPmjETejHtxX1BDY+l+L5vLgAdC/jGmHAzrB97HTh7NhqFICjwbyblqGjVy3UXAu26ofVGMWC5TSXcKFLIC3WYCL2FXJueC4EgiWfCUsKJOL2g9tskl8pm0Bx4Kgyu3Xs50v+KkvAEVJ+UHvU7KL4U1jsPAnFQpk3Qmk63uKaVuUNVYc+asqgqrdAMI7ukGELm3G0AAd71KA6LPIEChvCEVJVahHldcJfUOiFVid2FKzZw0+0Erl07MHeINTuSjKzK5+PCqULB5XbeZ28gR9JorAVfuDF6dY6QNjY21JGFVe1otpK2CplpQtzpKKnXq5TfbSSixO40y1OVpXPfPyv2eF7LqNwoq8j6A8crcuZesNBE304VEVwI01KrKj8qzIzdgJz4OInEnXSqYx/v4hiyfzDtk+YTutMvxrdshxJzTJ3THL/k2x+25Fe8PqswDqmHPPKBa7J0HVMthg3CVI64UCeZ9ILjH+0DkXu8DAdzFGuxgcbPLKCYLUev2pWRW4PF/Im09dPPVuRFD5BPdadAeK3rAs8mpGBKmH/ABLeiS7iarCqtMxCG4XcKSc4oZ2TxwZOUbuYayIJn05uIJqEwV/KCl/zEIoXc880thft9iPObgC6n4/SrU4/erpF4nVSWGOTrrZXF1faagxN62noTC3FmVLLszaaR7IeUtGUJnovqWrCqs0lEguKejQOTeutm37JUZjOd3xRp8mNcj+YTuauuJ5aFoPpMrRHvMZNbMTekGzwlS+W779WHOXD5nTlVQQ+OQ7u5iV9R+Fw/2ogf3PWZuhBVq7UtvB5SXHs/o7k4loyvMzu5sMZrvsWTJRKezoCKbjxlJs/yuaRoWtvRVQmfi0HwxFc1lVuRrpPOpaB4sEaiZjmXSfWV6FpbOJPtM8Dq6xsQzXeC0gP4aabPgrWZDEF2Tor1tgi1BLrMims7kUmW6EU9nK+mGsGQqb6ebKibLMaZjY5hZ06hUsAnHLi8a6UJiHzO6vL021WWNeG0pg+KmjEK0z4yVI0RqiZCIe+LMrjFOiW+tke82Cj1gWUQtbVCK1mXkC7WpqTtZrM1suv2GOwdHWzvuRI1CIW3pKGfGizGzNvX6otakXlecmiwvVkwBf1OThmLZlTUmmknHXOaMZZ2RpMRiPa6R67Z8fgkO1wKnzFy3WVtZjFy3233h2XjcbcND3WzFZke6g73tOcwSgYaNZwrRwvJwtAl06a384SEnfKgnfHm7HdzkC26NbgWjW1DBIRAcqgqGpt0MT6QZnkgzPJEIPJEwKhiaiKOTJq+u0n1d2WZIeKq/FD7cE55Y3hztz1vZVgvSSSvGCJ8gEu13ogyrktjh/hjhcgx/JuEoIqlwFB4eQoanrPABnvDOrlDrICuk0APmNWBj3mjaXKFZYekiWJGWThQU62cib4VMtH7YK6jAs6uiteIKHDMHOqmdzmCLiQGDzkbC0UImmoiEh1QFgn9H26FOQl2Z3AojF7cmFF2R8DhLCKZZZjpe7mYr07GeXAY8JpkElZsFZ7Vd1ABsPQzKmd2Wl04bKcejIQB7JV40limmCxOCwWEIZDgsHGQ+HibwaBcatStnmlNgAqe8+WI2m8kV8tFMFlo9PwY+Q4dq0wGtWUIuk81PDIBAnSYHMPZC2WkBVNzMx3IJ6+lyENpZ7Ooyc1Fwb4CrhNkH7lzT5gpcAW0I6BeqOFdW5frOqAUEw0Y+sY8znQ6ArYWJFj6zNjzRnQZ3foElsRO2Q6BacOBKJ8JpIWWmKiAuue4yBbUfFwW2WIEatsOAhnH+D+3YDmateoYDzk9rTaQHSBWThQQkn3FoyspmjFduFgy3j5oAkfo8FATp9iEj4QiwvqHVIlD4SdXB1b5hSjAEPMNkJOYsTLcEkGr4vcd4LAJqMxFL2J5jJpYBi7fdNt+ApV1eBg+6UmyuBisLd1EiiA7dHROsq8jkC5AGgXmcacFYyd/MCEYr3mZ6LTDU11Sxbk8TTQcU2A1DLKDKKY1FMtbyYrQWPR4L0kUczDJrmMNyQ5U+CjFrn0tDZ+VyNOhaWX5mtEccc6Y64ETonNk1DiqsTOvGQuVph/AN6yWx0Qluj7yPutxMzsyCLwpgFtpUC+xu9saACJVmmxZAuhpvMgStntdV685L1VfJ7QlaNmmkrfabgAPs/XyqFexG7JledRt5JoLjq8W2X42BW03wlHUMlOgyjUIxZ+aHVElB2asLVj1MVKsxeP7ootyDRHUl/GNEtTKrBoBZOMTsL+RKHSdv2dlUHO4aALCcy26rO4LDlbt0JmUPQ9kCxK5RPaC64lWmX90QML+GTKjisapN2uewqlXsrly24Ovu1VZU7d6QxQJPjexiVRuw2wG2AemcObFsESxiAV0erOqyJkmFnJEo5Od2JjOxXuuu3P70yaYW/ON4vvvRmjNujW71X2VcjvffZtzyX2bc8r9m3PxfZtz8v2f837Vx8//axuH/ssbh/7XGof+yxqH/scbg4dh/k3El3n+ZMXjI9t9kXInnyzgEScA9XFg+bc4c18gHKyokhut2zS7JQPChWQE8TrNeAyfiDY2DuzI58MjX/tAumjNzRrp3iBNo5nIZa/KazaTz5qRSaOULwJIomjES9pd+XFeqEO3KjjBTK7qzxW3Tu2RyvRtn0uaiDFigBZaQliQ7pWNGFkwB4puAbMaWwpeYy4tmvrCxNXiVo43ziRfanxKW5eUMtzayC8G9aVkyqiSxU1ycyRfKsuG2bJE1GdrBelK+dSZeTJoD3QI7oidoI0vPo8yUPUybafCeG3zGCr6zBEc/50dXyezXFtaOcoPjZvl1GHibZz1Frw4MRZsG+QLBaFAV1hrdqiqsBRoWgoRVp9cMidsMidsMiRuBxA1Dw6rjhqqVYD0b9wdaD8aH+ALtp+Kw0HRyaFWo9Tzcn671MLyatZ6EVycMHoNXpxCOQgJD8MDUuLiZMtLdYOPXaDENPuyNFcx4tMdIx5NgWdZYl7xg5lKJNPiUuCweUEyDpRdmvD6ZSXfPnz+lsT5mmzZYfGot23M+RbXWvoCF93siAO/H2hUo4GNtLzgFrMPJpKP5rBkrJo1Cos8ED50KPdGcUTAtdwHezE+EYJW1DfbWpcMQzNaw8FQ2GU2DhbzRvLUO1FksFIQ1NE6vLbV4zugqbIZnU5l4YLapTLyhsaGGdKwMawE7ze5EurEG0IiBNdFbBZBGNtobXAtANTROqyktqya1oVZdAlrEQe3aBCTrbCphlWDzmtAaoNrytKoyE4VaJSobF7hVNvN5ZMW9tPVry9rYWqgas7UqhKy7zVpbcDu1QdquC8UbuQvEG7kLtM1iNp4snTZVXtAf0E4l3qlZjamXNh0LUnCJr6VdS2wtVI3ZWqqdURtrazdAW6WTmmqz6hJt/aqRtYocQbBo59XQOAEZp9xYZU0YwOxzzr9RcJSLXfsoOK8CfHmQyKQnBMODfUgsmUmbI5zAnJnKFOyHFdbT+HRhTFli5DNpMOh3FuP2Oqu8WRiNkoJHHUgheNSBTNcqD1JqbZ0yFim16jjUEVfcs9VawzzB9oZNYA+V4ZBwa6HLKEcANlAtrUIuPeItNTaYv9hLk8t9oBI0tzJ/AVORaTXFAR9tzh/kCi9vCOGE+aclI51w0GC5jLVmoZDJWSs1S7MacJiIdZOVjzpzrsrd0fhqxv7epkyM9BDWrAvMnEA5xsNF9gY31tLTFjcBdoxJGfauHJVuUSVqsOaMiFi9iXS8oXE8BLDXUZeIEITIWcfMwAUNjW0Qgf1NDzxSaX8BWFlK8eyyTIAQzqcZJSQCQcx0MQXPGkgaGsch49hpzoDIwYpVM1cAGUetZcilrZGm1wSDe/RM17QA1plUA7XOCkCd3IFdgl2UglIG31XZ7ZxvjPUYufnz62OZdL4wZQbEZGzRtEaYDVoiuHItUUPjZIjI2g3MLB+hlTO7YGlbi63haTufXMPs24llNx3MFo30ymgGYcC2DBHPajNEPEtWueuwZFmzG+wZlwBvRlMR4EGzoM6WzzHyhVAV6wBgoa6RBHeTwBcZibSZs53aKFSMlJGtR8lK2U31A86ZzX6uyc+VGyu7stADRrzCSrAZUOlZU74DGcFuXctP2h4R7PRg3Z8mrB1QIv8gZtIogCh5bzXss3PsF3A5I50HftOu2EQUt3yFmY5Yp3flJqCYFHjJmc8aud5JWKQ7mUilzBwyL9D6+ULOSEYj6LzKFlKPQ2LZVMsUFABOvF+WjfYYuVQmvXIMEkukuyPIwia7UmH7wUF+LJZpGYcS9yZSiWhvJEAeRuqiNxztyeQS+2TSyFI6dYz2p5LIbLqzhWgmn0em0Z3oNqzAvshkJGP3jqTZbcRWNqCokkGDpYh9kWgYmVzpGEaLQta/cmafxxZS+e6q2cREHxCNJ7q6Kg+gwa/8KHBmXDSRMrpNsLoRfASWN3MJy7fYMvt978pMsVDsKx1k19A4wi0z8uXTwsFQWZGs6Enks+DLs3SsLB/rkvdl+i3TLx+d3NDozhRs7xZxZTrRLctkc0Y6U9oqzZrXF3LF1CgY48Qf6ZYluiun5YFhwiUy03kznekzosUQtFyWf0CUy5IVCvloPtvrrvY0GNNtgmPwQPGt/zloIxYNrzD6SrXGJ2qTkAp6UYhyLMBA1j3sqvsklyybifWaBUjlEZBpVI6HA5M7GOQuorsc2YRlOmXZaLfMyBm9pivpMR5hPJ40rUNZHekElzQNbsty4BYhHHXVcjwMyaTSCWgVK2ODvbEftBHcowO086QynYmkmTYLfS3QzlMZFaBKcEaEPqj2UolUply9aCo1EiXzZ5nK+KrkdgTJpNFnQA3K2gGyGVoUa8Rx5eeJ1pVyiyb7RIbLOqxtFFcYfd7G9lPe5MFY40p+lE/UG26BahbcT+TSrojuHHuK6ZVFwy11m1d3zkiDLWHzWdOM9ZSRidVIs3UbkElDFd2dTDXDG9Yek/qg5bZlRawQ3ultoQGtsNlvZNKus+W9McFZ5e6+Ns4ns05OjxpwuTMguuPXw+VhqLmWj6nug5Y8lkmDl5qu7jfSI3WdGdvQOLgiqnSbHcCNkXPvY79C6TTyCfC9VLfZPxcIZ1jBpfco9gDt2iIEvNaZW0kEcINiRjpmJsH9V6IrETPA3GGgE2bErOdWsUJ/XrVe4q5IFHrAzyHWr0KPmY52gQllT7Rg5HtHWKFOXFsIlr7ECv2jHIlzVo1HZoDnIvXpZKYnZaTTc+ZYU2yjMwHmJaFwtAkMPgUjkSxtBDK3gtr1zIOvkaya2j3FSPqQGVYO1ver9h4+zusl+6vYueX3Wvl/Aif/CZz4J3CsBrjS+LXATro9BoAnBcA9mVxhfgMeAm8A7deAE/BgIl0IytDaUmH+ZDwUzxQ7k+b8iXjKesI2zSgWMuU9Y6y7yrS9gyt49mr913Q2cW5yo+7/x80C2PClmLc3VwAfyBm5XjPXEhDByOcT+YJhbw5QjjUJFcvu1/Yvb9JpI7lyH/uWNu8st4hV5vZOuvmGxg1qiFXecsTeJ64ct+2fxM2bYJZTyOQaGuf8k3jgcXEly1BtUfOVGDMxMdw/7Rv/uf+EjjpLTNLO49gN/5vIlrdKW8/3a4nvwq1pYzJpOkVvaIzA4pefhIPtAfxhDY0dAXHsrZXMaKEnke4Fz9PNNNi3MN7Q2F5jzMqj+HwsAzYegJocNiIYfsx0zGxonA2L67yPcFXQCWlonIHlfQHjjGzWs0OX53dD4+Coa5VMIWnf4A7zBForwa3w0ZBwa8QCW6rDIpn9icJISHgsZ+R7zPiIaNUSnZyZL2Ry5rBqib0puzs8VwQ2GrW6fqKUtifNWI8JFpwZiWRnpn9gNLrCyKecZzXWguEBTpA9EhcyubwajeYL8dKxAUrpF5jBlH+AbMpcLJnJm8Oi0axTtbyZ7CqPucOj3nrEM8WCldaEaDTW3290JvpCoJM7zWEtZ00Xs5ZPm+pG4mbWTMftLZJ8cMTNRaN9qUQ0ljTyeef7kXRXBrxptY6b7TTBspF4vvAP4hidmT5wkEhhfg1xwJbJxbSR6kx0FzPFfDRb7EwmYtbC4bA3er6GUtYepVzIecFRMGWc5o2dzVjt6I4MtsuIgW31Z3nRrmI6bgDfbiSh+GwvHlj1GvlyvecE8JhKNwRELddhtAVGu5OZTiPp7KwaB+/eOiIdzUhhW3t7CC1s7mhDC5ta0cLWjrYmtLAFk2drc0cYI2zCJBsOY5INtbRjhM0dSGFLWxtOGELn2dLa1IIWtrSjW6W5tQ2thObWFnSekbYWdMxISwQdM9zRGhqFFEbQeg+Hm9CphtpaI2OQwta2jhBG2tTaik64pakNHTXSHsEkHGlta8FIW9oxRY40t7ejpeHWZrT6Q+FIByZqqLUZI20Kt2KlTThpCBu3CVOqUEcHJm6oA6eMUEc7pkahjnZMmUMdbdh8cboKdbRga9QSxkmbsSlHsPUNYeM24WrUjtVVO1ZX7ThzDrXjemCoPYItVRhbKpzVhdpxVhdqb8KVqq0D14JtHbj6tuG6b6itDeMYQm1Yu2prxdW3rRlnG21Yu2qL4GyyLYxxdaG2MLa+IVyZWztwZW7twKXc2o5ro9Y2vBSnjdZWnDZaW7BlbsGm3IzVRgQbN4KNG8KWCtv3Wzpw9W3B+tgWrN+ItOMsJ4zrR01YL9rU3oxJuakVM18INTXjEm7GdYWm5iZckcO4TtbU1BIai5DO6gDT5LZxSHEHmOq1j0TL0VHboy2trW3oqJhUW9GiZnSGEVBWTNQIWhQegRQhI7WjK9Deiixle0u0oznSgo7aUo8URYBlR9AqbUfXsD2MFiGnv7Pam1owMrTS0JppQ+uzDV2xthakPtsi0Y7mcAdSaW0RMKFuxpQIrbS2EFrUhKw8uje1gtluWwQt7wB1QauhFd3bWtuj4XC4CR21HR21DeSK7jeYPtyKiYU28dZmtAijnHC0NRzClAVt4a3odmxFtyO6D7cAh9zchJeHMHKrndHlbUGbc0s7MOe2diwQbo2gjboFE7fNShzdqC1o225pjYY7MB2mpRWdbYuVLdoqWtDdv6UZZNuETrvZSjuMBcKtGGttaUbHjVhxMbqOoOOGrYKhe2wLpswh6yEBpiFDoGAtaF/bgmkodI9AN0MzsOgQ2jqaO6LhcAg9tDV3WAVGt0Izuks0Wxbfgm6FZrTvawa+L4Q2zeY2K220tprb0HFbreciHcgRtBnjPpsxZWqx0sVkDIBQB7r1mzENYXWHZkxDoCdgzWAoDqG7cTOmicLoVEMgVUzUEDpqE3iUix40m5uiLa0hjHU0gaeDGEU3WYpGzy0wY0TEmvKje5Qlb0KPQRGrx6HHkAja7iJt6FldBOPCMTPPCNqSMXPdCNpWIuiJQARMBCKYmmNEmMqh3R66KGHQCE1tSG2GO9Bz63AH2veH0Y0XbsfJ2jEyTDEx9yth4CExdhhuAy8gIui023CyMEYWGo+WNUVDoeZQEyY2RrnoYSrcCl5eoIeacCt4TB7C1Ki1A13q1nZQ6iZMG7WiXUkY0wTglhJzpxZuwRS4BaMotHcKN0db2trRM5RwM3BfmPE8jB5GwmAYaWpGFzmCqWoEUx1MPwZjTBNW3tLWjh5owiHwAgo9RoVDSE2FOqwxBO0/Q+jxK9QOioWe/IbarbTRE1gANDVjbo5D6DlmqM2Kiyk4up+FLIvFFLzFelmF7imhFrR9hDDTtRDadEIRtIMMRTDZYQaVEHr+Hwpb79TQzjwUxlQf85AAMycIoUejEMZ/hJrQ/iOEeQoUGoaQDAThRjRbzJnRvkSuUDSSih1kFMDanMnRzRZvu9HCxdFovtgZ3cL+SgV8kxJ1lhHlZsWy2fG7bLbdTvYWPFvYX3h7AxoaG/3ELolkcuNcJrvYyBc26bcXwywBX51apL13TwBJECRBETTBECzBESzBEwIhEizBEhLBEjKhECqhEboVUkewxABiIDGIGGz9ZokhBEsMJYYRw4kRxEhiFDGaYIkxBEuMJcYR9cR4YgIxkZhETCamEFOJBqKRYIlpxHRiBjGTmEXMJpqIEBF2UooQzUQL0Uq0Ob/biQ5iDrEBwRJziXnEfGJDYgHBEguJjYhFxMbEJsSmxGbE5sQWxJYES2xFLCa2JrYhtiW2I7YnlhA7EDsSOxE7E7sQuxK7WantTrDEHsSexFiCJfYi9rb+jRJLiQWEQXQSMSJuUab1dxfRTfQQCWKZ9auXYIkkkSLSRIbIEsuJHJEnCpakSPQRK4h+YiWxD7EvsZ9T7v2JVcQBxIHEWOIg4mDiEOJQ4jDicOII4kjiKOJo4hjiWIIljiNWOzRLHE+c8B+tnkicRJxMnEKcaoWdRpxOnOHIzyTOIljibOIc69e5xHkES5xPXEBEiAuJi4iLiUsIlriUuIy4nGCJKwiWuPI/Or6KuJq4hlhDXEtcR1xP3EDcSNxE3EzcQtxK3GalcjtxB3EncRdxN3GP9fte4j7r3/uJB5xcHyQeIh4mHiEetX49RjxOPEE8+R9NP2X9fpp4hniWeK5ch+eJF4gXiZeIl4lXiFeJ5cRrxOvEG47sTWItcSnxFvE28c5/dP4u8R7xPvGBJfmQ+Ij4mPiE+NT69RnxOfEF8SXxFcESXxPfEN8S3xHfW5J1xA/Ej8RPxM/EfOIX4ldCJljiN+J34g/iz//o+y/ib2I9QZAkSZEsQZMMyZIcyZMCKZISKZMsUbkUUiU1UifryAHkQHIQOdiRDiGHksPI4eQIciQ5iBxFjibHOJKx5Diynhxv/ZpATiQnkZPJKeRUssGRN5LTyOmkRswgZ5KzXHnNJpvIEBkmI2Qz2UK2km1kuyXtIOeQc8gNyLnkPHI+uSG5gFxIbkQuIjd2xd2E3JTcjNyc3ILcktyKXExuTW5DbktuR27vYpaQO5A7kjuRO5O7kLuSu5G7k0uJPcg1xJ7kXuTeZJRcShpkJxnz1D5OxkmT7CK7yR4yQS4je8kkmSLTZMZDgStLLidzZJ4skEWyj1xB9pMryX3Ifcn9yP2rWHCtIg8gDyQPIg8mDyEPJQ8jDyePII8kjyKPJo8hjyWPI1eTx5Pr15/giXsiuZw4iTyZPIU8lTyNPJ08gzyTPIs8mzyHPJdcQJxHnk9eQF5IXuTL72LyEvJS8jLycvIK8kryKvJq8gziGnKNr51Z4lryOvJ68gbyRvIm8mbyFvJW8jbydvIO8k7yLvJucj5xD3kveR95P/kA+SD5ELROLPEw+Qj5KPkY+Tj5BPkk+RT5NPkM+Sz5HLkl8Tx5NXk1+QL5InkG8RL5MvmKK4VXydfId4mx5OvkG+Sb5FryLfJt8h3yXfI98n3yA/JD8iPyY/IT8lPyM/Jz8gvyS/Ir8m7ya3I+8Q35Lfkd+T25hlxH/oAokf/6kfyJ/Jn8hfyV/I38nfyD/JO8hfiL/JtcTxIUSVEUTTEUS3HUEwRPCZRISdQaUqYUCp+qSmmUTtVRA6iB1CBqMDWEGkoNo4ZTI6iR1ChqNDWGGktNIcdR9dR4agI1kZpETaamUGuIqVQD1UhNo6ZTM6iZ1CxqNiWTTVQoIDeWCFMRqpnSqE3JFqqVWk60Ue1UBzWH2oCaS82j5lMbUguohdRG1CJqY2oTalNqM2pzagtqS2orajG1NbUNtS2lEdtR21NLqB2oHamdqJ2pXahdqd2o3ak9qD2pNeRegSXwXntTUWopZVCdVIyKUybVRXVTPVSCWkb1UkkqRaWpDJWlllM5Kk8VqCLVR62g+qmV1D7UvtR+1P7UKuoA6kDqIOpg6hDqUOow6nDqCOpI6ijqaOoY6ljqOGo1dTx1AnUidRJ1EnUydQp1KnUadTp1BnUmddY/LKn/Ops6hzqXOo9aR5xPXUBdSF1EXUxdQr1BXkpdRl1OXUFdSV1FXU1dQ62hrqWuo66nbqBupG6ibqZuoW6lbqNup+6g7qTuou6m7qHupe6j7qceoB6kPiUeoh6mHqEepRRiDQHs/jHqceoJ6knqKepp6hlqDfks9Rz1PPVCzWV/kXqJepl6hXqVeo16nXqDepNaS71FvU29Q71LvUe9T31AfUh9RH1MfUJ9Sn1GfU59QX1JfUV9TX1DfUt9R31PraN+oH6kfqJ+pn6hfqV+o36n/qD+pP6i/qbWUwRN0hRN0wzN0hzN0wIt0hIt0wqt0hqt03X0GnKN1ccG0ANpmRxED6aH0P+b1v3XUHoYPZweQY+kR9Gj6TH0WHocXU+PpyfQE2mSmERPoifTU+hFxFS6gW6kp9HT6Rn0THoWPZtuokN0mI7QzXQL3Uq30e10Bz2H3oCeS8+j59Mb0gvohfRG9CJ6Y3oTelN6M3pzegt6S3orejG9Nb0NvS29Hb09vYTegd6R3onemd6F3pXejd6d3oPek96a2ovem47SS2mD7qRjdJw26S66m+6hE/QyupdO0ik6TWfoLL2cXkPk6DxdoIt0H72C7rc0tpLeh96X3o/en15FH0Af+H9YY/DrIPpg+hD6UPow+nD6CPpI+ij6aPoY+lj6OHo1fTx9An0ifRJ9Mn0KfSp9Gn06vZw4gz6TPos+mz6HPpc+jz6fvoC+kL6Ivpi+hL6Uvoy+nL6CvpK+ir6avoZeQ19LX0dfT99A30jfRN9M30LfSt9G307fQd9JrybuoodQd9P30PfS99H30w/QD9IP0Q/Tj9CP0o/Rj9NP0E/ST9FP08/Qz9LP0c/TL9Av0i/RL9Ov0K/Sr9Gv02/Qb9Jr6bfot+l36Hfp9+j36Q/oD+mP6I/pT+hP6V7yM/pz+gv6S5qlv6K/pr+hv6W/o7+n19EN1A/0j/RP9M/0L/Sv9G/07/Qf9Pr1f9J/0X/T62mCIRmKoRmGYZlnKY7hGZkUGJGRGJlRGJXRGJ2pYwYw/2+0y/8Xr4HMIGYwM4QZygxjhjMjmJHMKGY0M4YZy4xj6pnxzARmIjOJmcxMYaYyDUwjM42ZzsxgZjJrqVnMbKaJCTFhJsI0My1MK9PGtDMdzBxmA2YuM4+Zz2zILGAWMhsxi5iNmU2YTZnNmM2ZLZgtma2YNdRiZmtmG2ZbZjtme2YJswOzI7MTszOzC7MrsxuzO7MHsyezF7M3E2WWMgbTycSYOGMyXUw308PsRCaYZUwvk2RSTJrJMFlmOZNj8kyBKTJ9zAqmn1nJ7MPsy+zH7M+sYg5gDmQOYg5mDmEOZQ5jDmeOYI5kjmKOZo5hjmWOY1YzxzMnMCcyJzEnM6cwpzKnMaczZzBnMmcxZzPnMOcy5zHnMxcwFzIXMRczlzCXMpcxnxL11HrqcuYK5krmKiZCXM1cw6wm1jDXMtcx1zM3MDcyNzE3M7cwtzK3ErcxtzNriDuYO5m7mLuZe5h7mfuY+5kHmNnEg8xDzMPMw8wjzAD6UWYY8RjzOPME8yTzFPM08wzzLPMc8zzzAvMi8xLzMvMK8yrzGvM68wbzJrOWeYt5m3mHsX31GvJR5l3mXeZ46j0GjJvvMx8wHzIfMuvKXukogmc+YmTyY+YT5lPmM+Zz5gvmS+Yr5mvmG+Zb5jvme2Yd8wOjMzrzI/MT8zPzy/9v+8mvzG/M78wfzJ/MX8zfzKvkeoZgSZZiaZZhWZZjeVZgRVZiZfZCSmFVVmN1to4dwA5kB7GD2SHsUHYYO5wdwY5kR7Gj2THsWHYcW8+OZyewE9lJ7GR2CjuVbWAb2WnsdHYGO5Odxc5mm9gQG2bDbIRtZpvZFraVbWPb2Q52DrsBO5edx85nN2QXsAvZjdhF7MbsJuym7Gbs5uwW7JbsVuxidmt2G3Zbdjt2e3YJuwO7I7sTuzO7C7sruxu7O7sHuye7F7s3u5yIsktZg+1kr6FibJw12S62m+1hE+wytpdNsik2zWbYLLuczbF5tsAW2T52BdvPrmT3Yfdl92P3Z1exB7AHsgexB7OHsIeyh7GHsYezR7BHskexR7PHsMeyx7Gr2ePZE9gT2ZPYk9lT2FPZ09jT2TPYM9mz2LPZc9hz2fPY89kL2AvZi9iL2UvYS9nL2MvZK9gr2avYq9lr2DXsPfS17HXs9ewN7KHMjexN7M3sLeyt7G3s7ewd7J3sXezd7D3svex97P3sA+yD7EPsw+wj7KPsY+zj7BPsk+xT7NPsM+yz7HPs8+wL7IvsS+zL7Cvsq+xr7OvsG+yb7CByLfsW+zb7Dvsu+x77PvsB+yH7Efsx+wn7KfsZ+zn7Bfsl+xX7NfsN+y37Hfs9u469kfiB/ZH9if2Z/YX9lf2N/Z39g/2T/Yv9m13PEhzJURzNMZxAsRzH8ZzAiZzEyZzCqZzG6VwdN4AbyA3iBnNDuKHcMG44N4JTCIUYyY3iRnNjuLHcOK6eG89N4CZyk7jJ3BRuKreGWEM0cI3cNG46N4Obyc3iZnNNXIibTYS5CNfMtXCt3DVkG9dA5eh2roObw23AzeXmcfO5DbkF3EJuI24RtzG3Cbcptxm3ObcFtyW3FbeY25rbhtuW247bnlvC7cA9zzzP7MitX79+/U7cztwu3K7cbtzu3B7cntxe3N5clFvKGVwnF+PinMl1cd1cD3cBmbD+JLhl3JtMLweuJPc2Q1pXiltDprk0V/E1Ga50HU+9z7zHZLnl3AnUVGod+QFzIpXhgKfKcXkuzxW4D5nSdTxFMwPoIsc4MVjGjvMocyLFMX3cCq7fulZy+3D7cvtx+9L7c/tzq7hV3AHcgdxB3MHcIdyh3GHcYdzh3BHckdxR3NHcMdyx3HHcau547gTuRO4k7mTuFO5U7jTudO4M7kzuLO4s7mzuHO5c7lzuPO7/tt/59/r3+j93nc9dwF3IXcRdzF3CXcpdxl3OXcFdyV3FXc1dw63hruVeJa/jrudu4G7kbuJu5m7hbuVu427n7uDu5O7i7ubu4e7l7uPu5x7gHuQe5B7iHuYe4R7lHuMe557gnuSe4p7mnuGe5Z7jJjHPcy9wL3IvcS9zr3Cvcq9xr3NvcG9ya7m3uLe5d7h3ufe497kPuA+5j7iPuU+4T7nPuM+5L7gvua+4r7lvuG+577jvuXXcD9yP3I/cT9zP3C/cr9xv3O/cH9yf3F/c39x6juBJnuJpnuFZnuN5XuBFXuAlXuYV52+V13idr+MH8AP4gfwgfjA/hG9nh/LD+OH8HHYEP5IfxY/mx/Bj+XH8aqKeH89P4Cfyk/jJ/BR+Kt/AN/LT+On8DH4mP4ufzTfxIT7MR/hmvoVv5dv4dr6Dn8NvwM/l5/Hz+Q35BfxCfiN+Eb8xvwm/Kb8Zvzm/Bb8lvxW/mN+a34bflt+O355fwu/A78jvxO/M78Lvyu/G787vwe/J78XvzUf5pbzBd/IxPs6bfBffzffwCX4Z38ufSCT5FL+QSfMZPssv56eROf5gIs8X+CK/G9HHr+D7+ZX8Pvy+/H78/vwq/gD+QP4g/mA+yx3CH8ofxh/OH8EfyR/FH80fwx/LH8ev5o/nT+BP5E/iT+ZP4U/lT+NP58/gz+TP4s/mz+F3YM7lz+XP48/nL+Av5C/iL+Yv4S/lL+Mv56/gr+Sv4q/mr+HX8Nfy1/HX89fzN/A38jfxN/O38Lfyt/G383fwd/J38Xfz9/D38vfx9/MP8A/yD/EP84/wj/KP8Y/xj/NP8E/yT/G7MU/zz/DP8s/xz/Mv8C/yL/Ev818Qr/Cv8q/xr/Nv8G/ya/m3+Lf5d/h3+fes633+A/5D/iP+Y/4T/lP+M/5z/gv+S/4r/mv+G/5b/jv+Kvp7fh3/A/8j/xP/Mz+P+oXvYY5gf+V/43/n/+D/5P/i/+bX84RACpRAC4zACpzAC4IgCpIgC4qgCpqgC3XCAGGgMEgYLAwRhgrDhOHCCGGkMEoYLYwRxgrjhHphvDBBmChMEiYLU4SpQoPQKEwTpgszhJnCLGG20CSEhLAQEZqFFqFVaBPahQ5hjrCBMFeYJ8wX5gsbCguEhcJGwiJhY2ETYVNhM2FzYQthS2ErYbGwtbCNsK0A7qG3E7YXrmaXCDsIOwo7CQcxOwu7CLsKuwm7C3sIewp7CXsLUWGpYAidQkyIC6bwKNMldAvrSXD1CAlhmdArJIWUkBYyQlZYLiwXckJOyAt5oSAUhKJQFPqEPmGFsELoF/qFlcJKYR9hH2FfYV9hP2E/YX9hlXCAcKBwkHCwcIhwqHCYcLhwhHCkcJRwtHCMcKxwnLBaOF44QThROEk4WThFOFU4TThdOEM4UzhLOFs4RzhXOE84X7hAuFC4SLhYuES4VLhMuFy4QrhSuEq4WrhGWCNcK1wnXC/cINwo3CTcLNwi3CrcJtwu3CHcKdwl3C3cI9wr3CfcLzwgPCg8JDwsPCI8KjwmvGnN1B4XnhCeFJ4SnhaeEZ4VfqKeE54XXhBeFF4SXhZeEV4VXhNeF94Q3hTWCm8JbwvvCO8K7wnvCx8IHwofCR8LnwifCp8JnwtfCF8KXwlfC+vIdeQ3wrfCd8L3wjrhB3YpvZQGz+5uJNaxS2nw50ZiLb2WXkr/IPwo/CT8LPwi/Cr8JvzC/i78Ifwp/CX8LawXCJEUKZEWGZEVOZEXBVEUJVEWFVEVNVEX68QB4kBxkDhYHCIOFYeJw8UR4kf0SHGUOFocI44Vx4n14nhxgihQS6iJ4jdkhJgkThaniKuJqaLKNYiN4jRxujhDnCmuJlYTs8TZYpMYEsNiRGwWW8RWsU0cwLWLHeIccQNxrjhPnC9uKC4QF4obiYvEjcVNxPup4dym4mbi5uLmokJsIW4pbiUuFrcWtxG3FbcT7yO3F5eIO4g7ijuJO4u7iLuKu4m7i3uIa4g9xb3EvcWouFQ0xE4xJsZFU+wSu8UeMSEuE3vFpJgS02JGzIrLxZyYFwtisXyBOdv7jH/GBmZ5RRE8++8TpxCt5AqxX1wp7iPuK+4n7i+uEg8Q53EHigeJB4uriUPEQ8XDxMPFI8QjxaPEo8VjxGPF48TV4vHiCeKJ4kniyeIp4qniaeLp4hnimeJZ4tniOeK54nni+eIF4oXiReLFIpjPXiJeKoIZ7WXi5eIV4pXiVeLV4jXiGvFa8TrxevEG8UbxJvFm8RbxVvE28XbxXiHOnUrcIV7P3yn2c3eJd4l3i3eL94j3iPeK94r3ifeJ94v3iw+ID4gPig+KD4kPi4+Ij4qPiY+LT4hPik+JT4vPiM+Kz4nPiy+IL4oviS+Lr4iviq+Jr4tviG+Ka8W3xP/bo/+/17/Xv9e/17/Xv9e/17/Xv9e/17/Xv9e/17/Xv9e/17/Xvxfuelt8R3xXfE98X/xAvIr7UPxI/Fj8RPxU/Ez8XPxC/FL8SvxK/Fr8RvxW/Fb8Tvxe/F5cJ/4g/ij+JP4s/iz+Iv4q/ir+Jv4u/i7+If4p/in+Jf4t/i2uFwmJkEiJkmiJkRiJlTiJk3jpBk6QROlGTpJkSZFU6WZOk3SpThogDZQGSYOlIdJQaZg0XBohsexIaZQ0WhojjZXGSfVSvTRemiBNlCZJk6Up0lSpQWqUpknTpRnSgeRMaZY0W2qSQlJYikjNUovUKrVJ7VKHNEfaQJorzZPmSxtKC6QF0kJpDL2RtEjaWNpE2lTaTNpc2kLaUtpKWizp7NbSNtK20nbS9tISaQfpC25HaSfpYmpnaRdpV2k3aXdpD2lPaS9pbykqLZUMqVOKSXHJlLqkbqlHSkjLpF4pKaWktJSRstJyKSflpYJUlPqkFVK/9Be3UtpH2lfaT9pfWiUdIB0gHSgdJB0sHSwdIh0iHSodKh0mHS4dIR0pHSUdLR0jHSsdJ62WjpdOkE6UTpJOlk6RTpVOk06XzpDOlM6SzpbOkc6VzpPOly6QLpQuki6WLpEulS6TLpeukK6UrpKulq6R1kjXStdJ10s3SDdKN0k3S7dIt0q3SRuzt0t3SHdKd0l3S/dI90qt/H3S/dIDUjv/oNTBPyQ9LD0sPSI9Kj0qPSY9Lj0uPSE9KT0lPSU9LT0jPSs9Jz0nPS+9IL0ovSi9JL0svSy9Ir0qvSa9Lr0uvSG9Ka2V1kpvSW9L8/h3pHel96T3pQ+kD6WPpI+lT6RPpc+kz6UvpC+lr6SvpW+kb6XvpO+lddIP0o/ST9LP0i/Sr9Jv0u/SH9Kf0l/S39J6iZBJmZJpmZFZmZN5WZBFWZJlWZFVWZN1uU4eIA+UB8mD5SHyUHmYPFweIY+UR8mj5THyWHmcXC+PlyfIE+WJ8iR5sjxZXsZPkafIU+UGuUFO843yNHmaPF2eIc+UZ8qz5NlykxySw3JYjsjNcovcKrfJ7XKHPEfeQN5AnivPk+fLG8oL5AXyQnkjeZG8SN5Y3kTeVN5U3kzeXN5C3kLeUt5KXixvLW8jbytvJ28vL5F3kHeUd5J3lneWd5F3lXeTd5N3l/eQ95T3kveW95aj8lLZkPdmO+WYHJdNuUvukrvlHjnLJ+Rl8jK5V07KKTklp+WMnJWz8nI5J+flvFyQi3KfvELejOiXV8r7yPvK+8n7y6vkA+QD5YPkg+VD5EPlw+TD5SPkI+Wj5KPlY+Rj5ePk1fLx8gnyifJJ8snyKfKp8mny6fIZ8pnyWfLZ8jnyufJ58vnyBfKF8kXyxfIl8qXyZfLl8hXylfJV8tXyNfIa+Vr5Ovl6+Qb5Rvkm+Wb5FvlW+Tb5dvkO+U75Lvlu+R75Xvk++X75AflB+SH5YfkR+VH5Mflx+Qn5Sfkp+Wn5GflZ+Tn5efkF+UX5Jfll+RX5Vfk1+XX5DflNea38lvy2/I78rvye/L78gfyh/JH8sfyJ/Kn8mfy5/IX8pfyV/LX8jfyt/J38vbxO/kH+Uf5J/ln+Rf5V/k3+Xf5D/lP+S/5bXi8TCqlQCq0wCqtwCq8IiqhIiqwoiqpoiq7UKQOUgcpZ/CBlsDJEGaoMU4YrI5SRyihltDJGGauMU+qV8coEZaKyAzNJmaxMUaYqDUqjMk2ZrsxQZioX8bOU2UqTElLCSkRpVlqUVqVNaVc6lDnKBspcZZ4yX9lQWaAsVDZSFikbK5somyqbKZsrWyhbKlspi5WtlW2UbZXtlO2VJcoOyo7KTsrOyi7Krspuyu7KHsqeyl7K3kpUWaoYSqcSU+KKqXQp3UqPklCWKb1KUkkpaSWjZJXlSk7JKwWlqPQpK5R+ZaWyj7Kvsp+yv7JKOUA5UDlIOVg5RDlUOUw5XDlCOVI5SjlaOUY5VjlOWa0cr5ygnKCcqJyknKycopyqnKacrpyhnKmcpZytnKOcq5ynnK9coFyoXKRcrFyiXKpcplzNX65coVypXKVcrVyjrFGuVa5TrlduUG5UblJuVm5RblVuU25X7lDuVO5S7lbuUe5V7lPuVx5QHlQeUh5WHlEeVR5THleeUJ5UnlKeVp5RnlWeU55XXlBeVF5SXlZeUV5VXlNeV95Q3lTWKm8pbyvvKO8q7ynvKx8oHyofKR8rnyifKp8pnytfKF8qXylfK98o3yrfKd8r65QflB+Vn5SflV+UX5Xb+N+U35U/lD+Vv5S/lfUKoZIqpdIqo7Iqp/KqoIqqpEqqrCqqqmqqrtapA9SB6iB1sDpEHaoOU4erI9SR6ih1tDpGHauOU+vVHDtenaBOVCepk9Up6lS1QW1Up6nT1RnqTHWWOlttUkNqWI2ozWqL2qq2qe1qhzpH3UCdq85T56sbqgvUhepG6iJ1Y3UTdVN1M3VzdQt1S3UrdbG6tbqNuq26nbq9ukTdQd1R3UndWd1F3VXdTd1d3UPdU91L3VuNqktVQ+1UY2pcNdUutVvtURPqMvUBvldNqik1rWbUrLpczal5taAW1T51hfo436+uVPdR91X3U/dXV6kHqAeqB6kHq4eoh6qHqYerR6hHqkepR6vHqMeqx6mr1ePVE9QT1ZPUk9VT1FPV09TT1TPUM9Wz1LPVc9Rz1XPV3Zjz1PPVC9QL1YvUi9VL1EvVy9TL1SvUK9Wr1KvVa9Q16rXqder16g3qjepN6s3q0/wt6q3qbert6h3qnepd6t3qPeq96n3q/eoD6oPqQ+rD6iPqo+pj6uPqE+qT6lPq0+oz6rPqc+rz6gvqi+pL6ofUy+or6qvqa+pr6uvqG+qb6qv8WvUt9W31HfVd9T31ffUD9UP1I/Vj9RP1U/Uz9XP1C/UL9Uvr+kr9Sv1a/Ub9Vv1OfY3/Xl2nrlN/UN/mf1R/Un9Wf1F/VX9Tf1f/UP9U/1L/VterUeZ9ntBIjdJojdFYjdU4jbcuQRM1SZM1WVM0VdM0XavTBmgDtUHaYG2INlQbpg3XRmgjtVHaaG209jX/NT9GG6ON1cZp9dp47Xt+gjZRm6hN0iZrU7R1/FStwboatWnaD/x0bYY2U5ulzdaatJAW1iJas9aitWptWrvWrnVY1xxtjraBNlebp83XNtQWaAu0n/if+IXaQm0jbZG2sbaJtqm2mba5toW2pbaVtljbWttG21bbTvuZ315bou2g/cLvqO2k7aztou2q7abtru2h7antpe2tRbWlmqF1ajEtrpmaqXVpXVq31qMltGUaIfRqvVpSS2kpLa1lNFrIasu1DemcltcKWlHr01Zo/dpKbR9tX20/bX9tlXaAFhEO1NqEg7SDtUO0Q7XDtMO1I7QjtaO0o7SjtWO0Y7XjtNXa8doJ2onaSdrJ2inaqdppGlgzcbp2hnamdpZ2tnaOdq52nna+tqFwgXahdpF2sXaJdol2qXaZdrl2hfY0caV2lXa1do1zrdGu1a7Trtdu0G7UbtJu1m7WbtFu1W7Tbtfu0O7U7tLu1u7W7tHu1e7T7tce0B7UHtQe0h7WHtEe1R7THtee0J7QntSe0p7WntGe1Z7Tntde0B4nXtResq6XtVe0V7XXtNe017U3tDe1tdpa7S3tbe0d7V3tXe097X3tA+1D7UPtI+1j7RPtU+1T7TPtc+0L7UvtK+1r7WvtG+1b7TvtO+17bZ32g/aj9qP2k/az9ov2q/ar9pv2u/aH9qf2p/aX9pe2ufC3tl4jdFKndFpndFbndF4XdEEXdUmXdUVXdVXXdF2v0wfoA/SB+iB9sD5EH6oP04frw/UR+kh9lD5aH6OP1cfq4/R6fbw+QZ+gT9Qn6dsLk/Up+lS9QW/Up+nT9On6DH2mPkufpc/Wm/SQHtYjerPeorfqrXqbvkRYIrTrHXqHPkffQJ+rz9Pn6fP1DfUF+kJ9I32RvrG+ib6Jvqm+mX4Qs7m+ub6FvqW+lb5Y31rfRt9W307fXl+i76DvqO+k76zvou+q76bvru+h76nvpe+tR/WluqF36jH9ICGum3qX3q336Al9mQ7W+vbqST2lp/WMntWX6zk9rxf0ot6nr9D79ZX6Pvq++n76/voq/QD9QP0g/WD9EP1Q/TD9cP0I/Uj9KP1o/Rj9WP04fbV+vH6CfqJ+kn6yc52in6qfpp+un6GfqZ+ln62fo98u3i6eq5+nn69foF+oX6RfrF+iX6pfpl+uX6FfqV+lX61fo6/Rr9Wv06/Xb9CPZ27Ub9Jv1m/R32Rv1W/Tb9fv0O/U79Lv1u/R79Xv0+/XH9Af1B/SH9Yf0R/VH9Mf15/Qn9Sf0p/Wn9Gf1Z/Tn9df0F/UX9Jf1l/RX9Vf01/X39Df1Nfqb+lv6+/o7+rv6e/rH+gf6h/pH+uf6J/qn+mf61/oX+pf6V/r3+jf6t/p3+vr9B/0H/Wf9J+tP7/ov1p/ftN/t/78of9p/flL/7v8Z71O1JF1lPWHrmOsP2wdZ/3h64Q6sU6qk+uUOrVOq9Pr6uoG1A2sG1Q3uG5I3dC6YXXD60bUjawbVTe6bkzd2LpxdfV14+sm1E2sm1Q3uW5K3dS6hrrGuml10+tm1M2sm1U3u66pLlQXrovU/cA2162lwd+lP/av6r/Buhvwf7DKurnOH+t+Csh/YO+jSv+32aVOCnastbRNuUPA33EuzrXUtda11bXXddTNqdugbm7d3Lp5dfPrNqxbULegbmHdRnWL6v4W/p8+7bKrCgSMojDlHa6Knv3aLVjY3aDo2I1idys2CooJItjd3d3dXdid2N3dxaxZaz7Pfv7GrqCKqqR4z8qqoqqqJhdnddVQTdWSq7O26ihQdVVPQaovN2cDNVQjNVYTNVUzNVcLtVQrtVYbtVU7xbm3Vwd1VCd1VqBbsHuwyrh0UVd1U3f1UE/1kpcjRCHqLV9nH4UqVGEKU1/1U7jC1V8DNFCDNFhDFKFIRWqoohSlYYpWtGIUo+EaoZEapdEao7Eap/GaoImapMmaoqmapumaoZmapdmao7maq3marwXycy7UIi3WEi3VMi3XCq3UKq3WGq3VOq3XBm3UJgU5N2uLtmqbtmuHvB07tUu7tUd7tU/7dUAHdUiHdUQ+jqM6puOK1Qmd1Cmd1hllcZzVOZ3XBV3UJV3WFV3VNV3XDd1UnG7ptu7oru7pvh7ooR7psZ7oqZ7p+X9e6KVe6bXe6K3e6b0+6KM+6bO+6Ku+6btiPX7I3/FTv/RbfxThjJcLrrjhjjseJMDx7/SJJ06cJCQRifEiCUkRYCQjOSlISSpSk4a0pCM9GchIJjLjjQ9ZyEo2spMDX3KSi9zkIS/5yE8BClKIwhShKMUoTglKUorS+OFPGcoSQDnK8zcVqEglKlOFqlSjOjWoSS1qU4dA6lKPIOrTgIY0ojFNaEozmtOClrSiNW1oSzva04GOdKIzwXShK93oTg960osQetOHUMLoSz/C6c8ABjKIwQwhgkiGEsUwoolhOCMYyShGM4axjGM8E5jIJCYzhalMYzozmMksZjOHucxjPgtYyCIWs4SlLGM5K1jJKlazhrWsYz0b2MgmNrOFrWxjOzvYyS52s4e97GM/BzjIIQ5zhKMc4zixnOAkpzjNGc5yjvNc4CKXuMwVrnKN69zgJnHc4jZ3uMs97vOAhzziMU94yjOe84KXvOI1b3jLO97zgY984jNf+Mo3vvODn/ziN3+Ix8Vczc3czcMSmMP+Mk9zWkJLZInNy5JYUpNhZsksuaWwlJbKUlsaS2vpLL1lsIyWyTKbt/lYFstq2Sy75TBfy2m5LLflsbyWz/JbAStohaywFbGiVsyKWwkraaWstPmZv5Wxshb/PwVYgJWz8vYPaNyFDEi7AgA=",
  "compat": "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"
};

// src/debug.ts
var cache = /* @__PURE__ */ new Map();
function loadMap(buildKey) {
  return __async(this, null, function* () {
    if (cache.has(buildKey)) return cache.get(buildKey);
    const b64 = WASM_SOURCE_MAP[buildKey];
    if (!b64) throw new Error(`No source map for build "${buildKey}"`);
    const gzipped = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ds = new DecompressionStream("gzip");
    const writer = ds.writable.getWriter();
    writer.write(gzipped);
    writer.close();
    const buf = yield new Response(ds.readable).arrayBuffer();
    const dv = new DataView(buf);
    const bytes = new Uint8Array(buf);
    const firstId = dv.getUint32(0, true);
    const funcCount = dv.getUint32(4, true);
    const numNames = dv.getUint32(8, true);
    const td = new TextDecoder();
    const names = [];
    let pos = 12;
    for (let i = 0; i < numNames; i++) {
      const len = bytes[pos++];
      names.push(td.decode(bytes.subarray(pos, pos + len)));
      pos += len;
    }
    const funcNames = [];
    for (let i = 0; i < funcCount; i++) {
      const idx = dv.getUint16(pos, true);
      pos += 2;
      funcNames.push(idx === 65535 ? null : names[idx]);
    }
    const entry = { firstId, funcNames };
    cache.set(buildKey, entry);
    return entry;
  });
}
var Debug = {
  /**
   * Resolves a list of wasm function indices to their cleaned symbol names.
   */
  decodeFuncIds: (funcIds, isCompatBuild) => __async(void 0, null, function* () {
    const buildKey = isCompatBuild ? "compat" : "default";
    const { firstId, funcNames } = yield loadMap(buildKey);
    return funcIds.map((funcId) => {
      const i = funcId - firstId;
      const name = i >= 0 && i < funcNames.length && funcNames[i] ? funcNames[i] : "(unknown)";
      return { funcId, name };
    });
  }),
  /**
   * Annotates a wasm stack trace string with resolved function names.
   *
   * Example input from Chrome:
   *   at http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[775]:0x74251
   *   at async blob:http://localhost:8080/53a863cc-7227-45cc-8594-ddbbf5257f20:317:28
   *
   * Example input from Firefox:
   *   @http://localhost:8080/esm/wasm/wllama.wasm:wasm-function[796]:0x7dfe2
   *       at wModuleInit/WebAssembly.promising/< (9b6a2acd-d909-44e2-b021-d42fb9087cfb:15:32) index.js:1433:45
   *
   * Example input from Safari:
   *   2441@wasm-function[2441]
   *       at wrapper (d746f19e-4523-4f36-ba06-d0969acc0b05:22:126009)
   *
   * Example output:
   *   wasm-func[775] (server_response::send)
   */
  decodeStackTrace: (stack, isCompatBuild) => __async(void 0, null, function* () {
    const re = /wasm-function\[(\d+)\]/g;
    const funcIds = [
      ...new Set([...stack.matchAll(re)].map((m) => parseInt(m[1])))
    ];
    if (funcIds.length === 0) return stack;
    const resolved = yield Debug.decodeFuncIds(funcIds, isCompatBuild);
    return resolved.map((r) => {
      if (r.name === "(unknown)") {
        return `    wasm-func[${r.funcId}] (unknown)`;
      }
      return `    wasm-func[${r.funcId}] (${r.name})`;
    }).join("\n");
  })
};

// src/utils.ts
var textDecoder = new TextDecoder();
var URL_PARTS_REGEX = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
var parseShardNumber = (fnameOrUrl) => {
  const matches = fnameOrUrl.match(URL_PARTS_REGEX);
  if (!matches) {
    return {
      baseURL: fnameOrUrl,
      current: 1,
      total: 1
    };
  } else {
    return {
      baseURL: fnameOrUrl.replace(URL_PARTS_REGEX, ""),
      current: parseInt(matches[1]),
      total: parseInt(matches[2])
    };
  }
};
var sortFileByShard = (blobs) => {
  const isFiles = blobs.every((b) => !!b.name);
  if (isFiles && blobs.length > 1) {
    const files = blobs;
    files.sort((a, b) => {
      const infoA = parseShardNumber(a.name);
      const infoB = parseShardNumber(b.name);
      return infoA.current - infoB.current;
    });
  }
};
var isMmproj = (blob) => __async(void 0, null, function* () {
  const META_NAME = "general.architecture";
  const META_VAL = "clip";
  const tmp = blob.slice(0, 128 * 1024);
  const header = yield tmp.arrayBuffer();
  const buf = new Uint8Array(header);
  const nameBytes = new TextEncoder().encode(META_NAME);
  const valBytes = new TextEncoder().encode(META_VAL);
  let offset = -1;
  outer: for (let i = 0; i <= buf.length - nameBytes.length; i++) {
    for (let j = 0; j < nameBytes.length; j++) {
      if (buf[i + j] !== nameBytes[j]) continue outer;
    }
    offset = i;
    break;
  }
  if (offset === -1) return false;
  if (offset + 8 * 4 + 4 > buf.length) return false;
  const view = new DataView(header);
  const valLen = view.getBigUint64(offset + 8 * 3, true);
  if (valLen !== /* @__PURE__ */ BigInt("4")) return false;
  for (let i = 0; i < valBytes.length; i++) {
    if (buf[offset + 8 * 4 + i] !== valBytes[i]) return false;
  }
  return true;
});
var absoluteUrl = (relativePath) => typeof document === "undefined" ? new URL(relativePath, self.location.href).href : new URL(relativePath, document.baseURI).href;
var padDigits = (number, digits) => {
  return Array(Math.max(digits - String(number).length + 1, 0)).join("0") + number;
};
var sumArr = (arr) => arr.reduce((prev, curr) => prev + curr, 0);
var isString = (value) => !!(value == null ? void 0 : value.startsWith);
var MMPROJ_FILE_NAME = "mmproj.gguf";
var prepareBlobs = (blobsInp) => __async(void 0, null, function* () {
  const blobs = [];
  let blobMmproj = null;
  for (const blob of blobsInp) {
    if (yield isMmproj(blob)) {
      blobMmproj = blob;
    } else {
      blobs.push(blob);
    }
  }
  sortFileByShard(blobs);
  const result = blobs.map((blob, i) => ({
    blob,
    name: `model-${padDigits(i + 1, 5)}-of-${padDigits(blobs.length, 5)}.gguf`
  }));
  if (blobMmproj) {
    result.push({
      blob: blobMmproj,
      name: MMPROJ_FILE_NAME
    });
  }
  return {
    llm: result.filter((f) => f.name !== MMPROJ_FILE_NAME),
    mmproj: blobMmproj ? { blob: blobMmproj, name: MMPROJ_FILE_NAME } : null,
    all: result
  };
});
var isSupportMultiThread = () => ((e) => __async(void 0, null, function* () {
  try {
    return "undefined" != typeof MessageChannel && new MessageChannel().port1.postMessage(new SharedArrayBuffer(1)), WebAssembly.validate(e);
  } catch (e2) {
    return false;
  }
}))(
  new Uint8Array([
    0,
    97,
    115,
    109,
    1,
    0,
    0,
    0,
    1,
    4,
    1,
    96,
    0,
    0,
    3,
    2,
    1,
    0,
    5,
    4,
    1,
    3,
    1,
    1,
    10,
    11,
    1,
    9,
    0,
    65,
    0,
    254,
    16,
    2,
    0,
    26,
    11
  ])
);
var isSupportExceptions = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      4,
      1,
      96,
      0,
      0,
      3,
      2,
      1,
      0,
      10,
      8,
      1,
      6,
      0,
      6,
      64,
      25,
      11,
      11
    ])
  );
});
var isSupportSIMD = () => __async(void 0, null, function* () {
  return WebAssembly.validate(
    new Uint8Array([
      0,
      97,
      115,
      109,
      1,
      0,
      0,
      0,
      1,
      5,
      1,
      96,
      0,
      1,
      123,
      3,
      2,
      1,
      0,
      10,
      10,
      1,
      8,
      0,
      65,
      0,
      253,
      15,
      253,
      98,
      11
    ])
  );
});
var isSupportJSPI = () => {
  return !!WebAssembly.Suspending;
};
var isSupportWebGPU = () => {
  return !!navigator.gpu;
};
var isSupportMem64 = () => {
  try {
    new WebAssembly.Memory({
      address: "i64",
      initial: /* @__PURE__ */ BigInt("1")
      // 1 page (64 KiB)
    });
    return true;
  } catch (e) {
    return false;
  }
};
var checkEnvironmentCompatible = () => __async(void 0, null, function* () {
  if (!(yield isSupportExceptions())) {
    throw new Error("WebAssembly runtime does not support exception handling");
  }
  if (!(yield isSupportSIMD())) {
    throw new Error("WebAssembly runtime does not support SIMD");
  }
});
var isFirefox = () => {
  return !!navigator.userAgent.match(/Firefox\/([0-9\.]+)(?:\s|$)/);
};
var GGUF_FILE_REGEX = /^.*\.gguf(?:\?.*)?$/;
var isValidGgufFile = (path) => {
  return GGUF_FILE_REGEX.test(path);
};
var isSafariMobile = () => {
  return !!navigator.userAgent.match(/Version\/([0-9\._]+).*Mobile.*Safari.*/);
};
var createWorker = (workerCode) => {
  const workerURL = URL.createObjectURL(
    isString(workerCode) ? new Blob([workerCode], { type: "text/javascript" }) : workerCode
  );
  return new Worker(workerURL, { type: "module" });
};
var cbToAsyncIter = (fn) => (...args) => {
  let values = [];
  let resolve;
  let reject;
  values.push(
    new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    })
  );
  fn(...args, (val, done, err) => {
    if (err) {
      reject(err);
      return;
    }
    resolve([val, done]);
    values.push(
      new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      })
    );
  });
  return function() {
    return __asyncGenerator(this, null, function* () {
      let val;
      for (let i = 0, done = false; !done; i++) {
        [val, done] = yield new __await(values[i]);
        delete values[i];
        if (val !== void 0) yield val;
      }
    });
  }();
};
var canUseAsyncFileRead = (compat) => isSupportJSPI() || compat;
var needCompat = () => !isSupportJSPI() || !isSupportMem64();

// src/workers-code/generated.ts
var LIBLLAMA_VERSION = "b12076-cab27bf";
var LLAMA_CPP_WORKER_CODE = "// Start the main llama.cpp\nlet wllamaMalloc;\nlet wllamaStart;\nlet wllamaAction;\nlet wllamaExit;\nlet wllamaDebug;\n\nlet Module = null;\nlet isCompat = false;\nlet lastStack = '';\nlet isAborted = false;\nlet hasMultithread = false;\n\n//////////////////////////////////////////////////////////////\n// UTILS\n//////////////////////////////////////////////////////////////\n\n// send message back to main thread\nconst msg = (data, transfer) => postMessage(data, transfer);\n\n// Convert CPP log into JS log\nconst cppLogToJSLog = (line) => {\n  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);\n  return !!matched\n    ? {\n        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),\n        text: matched[2],\n      }\n    : { level: 'log', text: line };\n};\n\nconst getHeapU8 = () => {\n  const buffer = Module.wasmMemory.buffer;\n  return new Uint8Array(buffer);\n};\n\nconst toSizeT = (num) => {\n  return isCompat ? Number(num) : BigInt(num);\n};\n\n// Get module config that forwards stdout/err to main thread\nconst getWModuleConfig = (_argMainScriptBlob) => {\n  var pathConfig = RUN_OPTIONS.pathConfig;\n  var pthreadPoolSize = RUN_OPTIONS.nbThread;\n  var argMainScriptBlob = _argMainScriptBlob;\n\n  isCompat = RUN_OPTIONS.compat;\n  hasMultithread = pthreadPoolSize > 1;\n\n  msg({\n    verb: 'console.debug',\n    args: [\n      `Multithread enabled: ${hasMultithread}, pthreadPoolSize: ${pthreadPoolSize}`,\n    ],\n  });\n\n  if (!pathConfig['wllama.wasm']) {\n    throw new Error('\"wllama.wasm\" is missing in pathConfig');\n  }\n  return {\n    noInitialRun: true,\n    print: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      msg({ verb: 'console.log', args: [text] });\n    },\n    printErr: function (text) {\n      if (arguments.length > 1)\n        text = Array.prototype.slice.call(arguments).join(' ');\n      if (text.startsWith('@@STACK@@')) {\n        lastStack = text.slice('@@STACK@@'.length);\n        return;\n      }\n      const logLine = cppLogToJSLog(text);\n      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });\n    },\n    locateFile: function (filename, basePath) {\n      const p = pathConfig[filename];\n      const truncate = (str) =>\n        str.length > 128 ? `${str.substr(0, 128)}...` : str;\n      if (filename.match(/wllama\\.worker\\.js/)) {\n        msg({\n          verb: 'console.error',\n          args: [\n            '\"wllama.worker.js\" is removed from v2.2.1. Hint: make sure to clear browser\\'s cache.',\n          ],\n        });\n      } else {\n        msg({\n          verb: 'console.debug',\n          args: [`Loading \"${filename}\" from \"${truncate(p)}\"`],\n        });\n        return p;\n      }\n    },\n    mainScriptUrlOrBlob: hasMultithread\n      ? argMainScriptBlob\n      : 'throw new Error(\"Multithreading is not enabled\")',\n    pthreadPoolSize: hasMultithread ? pthreadPoolSize : 0,\n    wasmMemory: hasMultithread ? getWasmMemory() : null,\n    onAbort: function (message) {\n      isAborted = true;\n      msg({ verb: 'signal.abort', args: ['abort', message, lastStack, null] });\n    },\n    onExit: function (code) {\n      isAborted = true;\n      const callstack = new Error().stack.toString();\n      msg({\n        verb: 'signal.abort',\n        args: ['abort', 'exit(' + code + ')', callstack, null],\n      });\n    },\n  };\n};\n\n// Get the memory to be used by wasm. (Only used in multi-thread mode)\n// Because we have a weird OOM issue on iOS, we need to try some values\n// See: https://github.com/emscripten-core/emscripten/issues/19144\n//      https://github.com/godotengine/godot/issues/70621\nconst getWasmMemory = () => {\n  let minBytes = 128 * 1024 * 1024;\n  let maxBytes = 4096 * 1024 * 1024;\n  let stepBytes = 128 * 1024 * 1024;\n  while (maxBytes > minBytes) {\n    try {\n      const wasmMemory = new WebAssembly.Memory({\n        initial: toSizeT(minBytes / 65536),\n        maximum: toSizeT(maxBytes / 65536),\n        shared: true,\n        address: isCompat ? undefined : 'i64',\n      });\n      return wasmMemory;\n    } catch (e) {\n      maxBytes -= stepBytes;\n      continue; // retry\n    }\n  }\n  throw new Error('Cannot allocate WebAssembly.Memory');\n};\n\n//////////////////////////////////////////////////////////////\n// HEAPFS PATCH\n//////////////////////////////////////////////////////////////\n\n/**\n * By default, emscripten uses memfs. The way it works is by\n * allocating new Uint8Array in javascript heap. This is not good\n * because it requires files to be copied to wasm heap each time\n * a file is read.\n *\n * HeapFS is an alternative, which resolves this problem by\n * allocating space for file directly inside wasm heap. This\n * allows us to mmap without doing any copy.\n *\n * For llama.cpp, this is great because we use MAP_SHARED\n *\n * Ref: https://github.com/ngxson/wllama/pull/39\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js\n *\n * Note 29/05/2024 @ngxson\n * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)\n * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c\n */\n\nconst fsNameToFile = {}; // map Name => File\nconst fsIdToFile = {}; // map ID => File\nlet currFileId = 0;\n\n// Patch and redirect memfs calls to wllama\nconst patchHeapFS = () => {\n  const m = Module;\n  // save functions\n  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;\n  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;\n  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;\n  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;\n  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;\n  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;\n\n  const patchStream = (stream) => {\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const ptr = Number(f.ptr);\n      stream.node.contents = getHeapU8().subarray(ptr, ptr + f.size);\n      stream.node.usedBytes = f.size;\n    }\n  };\n\n  // replace \"read\" functions\n  m.MEMFS.stream_ops.read = function (\n    stream,\n    buffer,\n    offset,\n    length,\n    position\n  ) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);\n  };\n  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;\n\n  // replace \"llseek\" functions\n  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {\n    patchStream(stream);\n    return m.MEMFS.stream_ops._llseek(stream, offset, whence);\n  };\n  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;\n\n  // replace \"mmap\" functions\n  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {\n    patchStream(stream);\n    const name = stream.node.name;\n    if (fsNameToFile[name]) {\n      const f = fsNameToFile[name];\n      const mmapPtr = f.ptr + toSizeT(position);\n      return {\n        ptr: mmapPtr,\n        allocated: false,\n      };\n    } else {\n      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);\n    }\n  };\n  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;\n\n  // mount FS\n  m.FS.mkdir('/models');\n  m.FS.mount(m.MEMFS, { root: '.' }, '/models');\n  // slot save files (action \"slots\"), moved to and from the main thread as bytes\n  m.FS.mkdir('/slots');\n};\n\n// Allocate a new file in wllama heapfs, returns file ID\nconst heapfsAlloc = (name, size, allocBuffer) => {\n  if (size < 1) {\n    throw new Error('File size must be bigger than 0');\n  }\n  const m = Module;\n  const ptr = toSizeT(allocBuffer ? m.mmapAlloc(size) : 0);\n  const file = {\n    ptr: ptr,\n    size: size,\n    id: currFileId++,\n  };\n  fsIdToFile[file.id] = file;\n  fsNameToFile[name] = file;\n  return file.id;\n};\n\n// Add new file to wllama heapfs, return number of written bytes\nconst heapfsWrite = (id, buffer, offset) => {\n  if (fsIdToFile[id]) {\n    const { ptr, size } = fsIdToFile[id];\n    const afterWriteByte = offset + buffer.byteLength;\n    if (afterWriteByte > size) {\n      throw new Error(\n        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`\n      );\n    }\n    getHeapU8().set(buffer, Number(ptr) + offset);\n    return buffer.byteLength;\n  } else {\n    throw new Error(`File ID ${id} not found in heapfs`);\n  }\n};\n\n//////////////////////////////////////////////////////////////\n// ASYNC FILE READ\n//////////////////////////////////////////////////////////////\n\nlet isAwaitReading = false;\nlet pendingReadPromise = null;\nlet pendingReadResolve = null;\nlet pendingReadReject = null;\n\nconst _stripModelsPrefix = (path) => path.replace(/^\\/?models\\//, '');\n\n// Called from EM_ASYNC_JS stub in wllama-fs.h (path is already a JS string)\nconst _wllama_js_file_read = async (path, offset, req_size, out_ptr) => {\n  const name = _stripModelsPrefix(path);\n\n  pendingReadPromise = new Promise((res, rej) => {\n    pendingReadResolve = res;\n    pendingReadReject = rej;\n  });\n  isAwaitReading = true;\n\n  postMessage({ verb: 'fs.read_req', args: [name, offset, req_size] });\n\n  let data;\n  try {\n    data = await pendingReadPromise;\n  } finally {\n    isAwaitReading = false;\n    pendingReadResolve = null;\n    pendingReadReject = null;\n  }\n\n  const bytes = new Uint8Array(data);\n  getHeapU8().set(bytes, out_ptr);\n  return toSizeT(bytes.length);\n};\n\n//////////////////////////////////////////////////////////////\n// MAIN CODE\n//////////////////////////////////////////////////////////////\n\nconst callWrapper = (name, ret, args, isAsync) => {\n  const fn = Module.cwrap(\n    name,\n    ret,\n    args,\n    isAsync ? { async: true } : undefined\n  );\n  return async (action, req) => {\n    // console.log(`Calling ${name} with action:`, action, 'and req:', req);\n    let result;\n    try {\n      if (args.length === 2) {\n        result = isAsync ? await fn(action, req) : fn(action, req);\n      } else {\n        result = fn();\n      }\n    } catch (ex) {\n      console.error(ex);\n      throw ex;\n    }\n    return result;\n  };\n};\n\n// re-entering the wasm while a call is suspended (JSPI / asyncify) corrupts its state, so only one call runs at a time and the rest wait in the queue\nlet wasmCallBusy = false;\nconst wasmCallQueue = [];\n\nconst runWasmCall = async (callbackId, fn) => {\n  if (isAborted) {\n    // the wasm is dead, fail fast instead of calling into it\n    msg({ callbackId, err: 'wllama has crashed, please reload the module' });\n    return;\n  }\n  if (wasmCallBusy) {\n    wasmCallQueue.push({ callbackId, fn });\n    return;\n  }\n  wasmCallBusy = true;\n  try {\n    await fn();\n  } finally {\n    wasmCallBusy = false;\n    if (isAborted) {\n      // do not touch the wasm again after it aborted; the main thread already rejected the queued tasks\n      wasmCallQueue.length = 0;\n    } else {\n      const next = wasmCallQueue.shift();\n      if (next) runWasmCall(next.callbackId, next.fn);\n    }\n  }\n};\n\nconst runAction = async (data) => {\n  const { args, callbackId } = data;\n  const argAction = args[0];\n  const argEncodedMsg = args[1];\n  try {\n    const inputPtr = await wllamaMalloc(toSizeT(argEncodedMsg.byteLength), 0);\n    // copy data to wasm heap\n    const inputBuffer = new Uint8Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      argEncodedMsg.byteLength\n    );\n    inputBuffer.set(argEncodedMsg, 0);\n    const outputPtr = await wllamaAction(argAction, inputPtr);\n    // length of output buffer is written at the first 4 bytes of input buffer\n    const outputLen = new Uint32Array(\n      getHeapU8().buffer,\n      Number(inputPtr),\n      1\n    )[0];\n    // copy the output buffer to JS heap\n    const outputBuffer = new Uint8Array(outputLen);\n    const outputSrcView = new Uint8Array(\n      getHeapU8().buffer,\n      Number(outputPtr),\n      outputLen\n    );\n    outputBuffer.set(outputSrcView, 0); // copy it\n    msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);\n  } catch (err) {\n    handleError(err);\n  }\n};\n\nfunction handleError(err) {\n  // If WASM already aborted, onAbort already sent signal.abort; skip to avoid\n  // re-reporting the resulting WebAssembly.RuntimeError as a JS exception.\n  if (isAborted) return;\n\n  const message = err ? err.message || String(err) : 'Unknown error';\n  const stack = err ? err.stack || String(err) : '';\n  msg({\n    verb: 'signal.abort',\n    args: ['exception', message, stack, err],\n  });\n}\n\nonmessage = async (e) => {\n  if (!e.data) return;\n  const { verb, args, callbackId } = e.data;\n\n  // fs.read_res arrives while wasm is JSPI-suspended; resolve the pending promise.\n  if (verb === 'fs.read_res') {\n    if (pendingReadResolve) {\n      pendingReadResolve(args[0]);\n    }\n    return;\n  }\n\n  // Guard: while awaiting a file read, reject any other incoming task.\n  if (isAwaitReading) {\n    if (callbackId) {\n      msg({\n        callbackId,\n        err: 'Worker is suspended waiting for file data (JSPI)',\n      });\n    }\n    return;\n  }\n\n  if (!callbackId) {\n    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });\n    return;\n  }\n\n  if (verb === 'module.init') {\n    const argMainScriptBlob = args[0];\n    const argUseAsyncFile = args[1];\n    try {\n      Module = getWModuleConfig(argMainScriptBlob);\n      Module.preRun = () => {\n        if (argUseAsyncFile) {\n          Module.ENV['USE_ASYNC_FILE'] = '1';\n        }\n      };\n      Module.onRuntimeInitialized = () => {\n        // async call once module is ready\n        // init FS\n        patchHeapFS();\n        // init cwrap\n        const pointer = isCompat ? 'number' : 'bigint';\n        // TODO: note sure why emscripten cannot bind if there is only 1 argument\n        wllamaMalloc = callWrapper('wllama_malloc', pointer, [\n          'number',\n          pointer,\n        ]);\n        wllamaStart = callWrapper('wllama_start', 'string', [], true);\n        wllamaAction = callWrapper(\n          'wllama_action',\n          pointer,\n          ['string', pointer],\n          true\n        );\n        wllamaExit = callWrapper('wllama_exit', 'string', []);\n        wllamaDebug = callWrapper('wllama_debug', 'string', []);\n        msg({ callbackId, result: null });\n      };\n      wModuleInit();\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.alloc') {\n    const argFilename = args[0];\n    const argSize = args[1];\n    const argAllocBuffer = args[2];\n    try {\n      // create blank file\n      const emptyBuffer = new ArrayBuffer(0);\n      Module['FS_createDataFile'](\n        '/models',\n        argFilename,\n        emptyBuffer,\n        true,\n        true,\n        true\n      );\n      // alloc data on heap\n      const fileId = heapfsAlloc(argFilename, argSize, argAllocBuffer);\n      msg({ callbackId, result: { fileId } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.write') {\n    const argFileId = args[0];\n    const argBuffer = args[1];\n    const argOffset = args[2];\n    try {\n      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);\n      msg({ callbackId, result: { writtenBytes } });\n    } catch (err) {\n      handleError(err);\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_read') {\n    const argFilename = args[0];\n    try {\n      const path = '/slots/' + argFilename;\n      const data = Module.FS.readFile(path);\n      Module.FS.unlink(path);\n      msg({ callbackId, result: data }, [data.buffer]);\n    } catch (err) {\n      msg({ callbackId, err: `cannot read slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_write') {\n    const argFilename = args[0];\n    const argData = args[1];\n    try {\n      Module.FS.writeFile('/slots/' + argFilename, argData);\n      msg({ callbackId, result: null });\n    } catch (err) {\n      msg({ callbackId, err: `cannot write slot file ${argFilename}: ${err}` });\n    }\n    return;\n  }\n\n  if (verb === 'fs.slot_remove') {\n    try {\n      Module.FS.unlink('/slots/' + args[0]);\n    } catch (err) {\n      // already removed\n    }\n    msg({ callbackId, result: null });\n    return;\n  }\n\n  if (verb === 'wllama.start') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaStart();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.action') {\n    await runWasmCall(callbackId, () => runAction(e.data));\n    return;\n  }\n\n  if (verb === 'wllama.exit') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaExit();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n\n  if (verb === 'wllama.debug') {\n    await runWasmCall(callbackId, async () => {\n      try {\n        const result = await wllamaDebug();\n        msg({ callbackId, result });\n      } catch (err) {\n        handleError(err);\n      }\n    });\n    return;\n  }\n};\n";
var OPFS_UTILS_WORKER_CODE = "let accessHandle;\nlet abortController = new AbortController();\n\nasync function openFile(filename) {\n  const opfsRoot = await navigator.storage.getDirectory();\n  const cacheDir = await opfsRoot.getDirectoryHandle('cache', { create: true });\n  const fileHandler = await cacheDir.getFileHandle(filename, { create: true });\n  accessHandle = await fileHandler.createSyncAccessHandle();\n  accessHandle.truncate(0); // clear file content\n}\n\nasync function writeFile(buf) {\n  accessHandle.write(buf);\n}\n\nasync function closeFile() {\n  accessHandle.flush();\n  accessHandle.close();\n}\n\nasync function writeTextFile(filename, str) {\n  await openFile(filename);\n  await writeFile(new TextEncoder().encode(str));\n  await closeFile();\n}\n\nconst throttled = (func, delay) => {\n  let lastRun = 0;\n  return (...args) => {\n    const now = Date.now();\n    if (now - lastRun > delay) {\n      lastRun = now;\n      func.apply(null, args);\n    }\n  };\n};\n\nconst assertNonNull = (val) => {\n  if (val === null || val === undefined) {\n    throw new Error('OPFS Worker: Assertion failed');\n  }\n};\n\n// respond to main thread\nconst resOK = () => postMessage({ ok: true });\nconst resProgress = (loaded, total) =>\n  postMessage({ progress: { loaded, total } });\nconst resErr = (err) => postMessage({ err });\n\nonmessage = async (e) => {\n  try {\n    if (!e.data) return;\n\n    /**\n     * @param {Object} e.data\n     *\n     * Fine-control FS actions:\n     * - { action: 'open', filename: 'string' }\n     * - { action: 'write', buf: ArrayBuffer }\n     * - { action: 'close' }\n     *\n     * Simple write API:\n     * - { action: 'write-simple', filename: 'string', buf: ArrayBuffer }\n     *\n     * Download API:\n     * - { action: 'download', url: 'string', filename: 'string', options: Object, metadataFileName: 'string' }\n     * - { action: 'download-abort' }\n     */\n    const {\n      action,\n      filename,\n      buf,\n      url,\n      options,\n      metadataFileName,\n      metadataAdditional,\n    } = e.data;\n\n    if (action === 'open') {\n      assertNonNull(filename);\n      await openFile(filename);\n      return resOK();\n    } else if (action === 'write') {\n      assertNonNull(buf);\n      await writeFile(buf);\n      return resOK();\n    } else if (action === 'close') {\n      await closeFile();\n      return resOK();\n    } else if (action === 'write-simple') {\n      assertNonNull(filename);\n      assertNonNull(buf);\n      await openFile(filename);\n      await writeFile(buf);\n      await closeFile();\n      return resOK();\n    } else if (action === 'download') {\n      assertNonNull(url);\n      assertNonNull(filename);\n      assertNonNull(metadataFileName);\n      assertNonNull(options);\n      assertNonNull(options.aborted);\n      abortController = new AbortController();\n      if (options.aborted) abortController.abort();\n      const response = await fetch(url, {\n        ...options,\n        signal: abortController.signal,\n      });\n      const contentLength = response.headers.get('content-length');\n      const etag = (response.headers.get('etag') || '').replace(\n        /[^A-Za-z0-9]/g,\n        ''\n      );\n      const total = parseInt(contentLength, 10);\n      const reader = response.body.getReader();\n      await openFile(filename);\n      let loaded = 0;\n      const throttledProgress = throttled(resProgress, 100);\n      while (true) {\n        const { done, value } = await reader.read();\n        if (done) break;\n        loaded += value.byteLength;\n        await writeFile(value);\n        throttledProgress(loaded, total);\n      }\n      resProgress(total, total); // 100% done\n      await closeFile();\n      // make sure this is in-sync with CacheEntryMetadata\n      await writeTextFile(\n        metadataFileName,\n        JSON.stringify({\n          originalURL: url,\n          originalSize: total,\n          etag,\n          ...metadataAdditional,\n        })\n      );\n      return resOK();\n    } else if (action === 'download-abort') {\n      if (abortController) {\n        abortController.abort();\n      }\n      return;\n    }\n\n    throw new Error('OPFS Worker: Invalid action', e.data);\n  } catch (err) {\n    return resErr(err);\n  }\n};\n";
var WLLAMA_EMSCRIPTEN_CODE = 'var Module=typeof Module!="undefined"?Module:{};var ENVIRONMENT_IS_WEB=!!globalThis.window;var ENVIRONMENT_IS_WORKER=!!globalThis.WorkerGlobalScope;var ENVIRONMENT_IS_NODE=globalThis.process?.versions?.node&&globalThis.process?.type!="renderer";var ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&self.name?.startsWith("em-pthread");if(ENVIRONMENT_IS_NODE){var worker_threads=require("worker_threads");global.Worker=worker_threads.Worker;ENVIRONMENT_IS_WORKER=!worker_threads.isMainThread;ENVIRONMENT_IS_PTHREAD=ENVIRONMENT_IS_WORKER&&worker_threads["workerData"]=="em-pthread"}var arguments_=[];var thisProgram="./this.program";var quit_=(status,toThrow)=>{throw toThrow};var _scriptName=globalThis.document?.currentScript?.src;if(typeof __filename!="undefined"){_scriptName=__filename}else if(ENVIRONMENT_IS_WORKER){_scriptName=self.location.href}var scriptDirectory="";function locateFile(path){if(Module["locateFile"]){return Module["locateFile"](path,scriptDirectory)}return scriptDirectory+path}var readAsync,readBinary;if(ENVIRONMENT_IS_NODE){var fs=require("fs");scriptDirectory=__dirname+"/";readBinary=filename=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename);return ret};readAsync=async(filename,binary=true)=>{filename=isFileURI(filename)?new URL(filename):filename;var ret=fs.readFileSync(filename,binary?undefined:"utf8");return ret};if(process.argv.length>1){thisProgram=process.argv[1].replace(/\\\\/g,"/")}arguments_=process.argv.slice(2);if(typeof module!="undefined"){module["exports"]=Module}quit_=(status,toThrow)=>{process.exitCode=status;throw toThrow}}else if(ENVIRONMENT_IS_WEB||ENVIRONMENT_IS_WORKER){try{scriptDirectory=new URL(".",_scriptName).href}catch{}if(!ENVIRONMENT_IS_NODE){if(ENVIRONMENT_IS_WORKER){readBinary=url=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,false);xhr.responseType="arraybuffer";xhr.send(null);return new Uint8Array(xhr.response)}}readAsync=async url=>{if(isFileURI(url)){return new Promise((resolve,reject)=>{var xhr=new XMLHttpRequest;xhr.open("GET",url,true);xhr.responseType="arraybuffer";xhr.onload=()=>{if(xhr.status==200||xhr.status==0&&xhr.response){resolve(xhr.response);return}reject(xhr.status)};xhr.onerror=reject;xhr.send(null)})}var response=await fetch(url,{credentials:"same-origin"});if(response.ok){return response.arrayBuffer()}throw new Error(response.status+" : "+response.url)}}}else{}var defaultPrint=console.log.bind(console);var defaultPrintErr=console.error.bind(console);if(ENVIRONMENT_IS_NODE){var utils=require("util");var stringify=a=>typeof a=="object"?utils.inspect(a):a;defaultPrint=(...args)=>fs.writeSync(1,args.map(stringify).join(" ")+"\\n");defaultPrintErr=(...args)=>fs.writeSync(2,args.map(stringify).join(" ")+"\\n")}var out=defaultPrint;var err=defaultPrintErr;var wasmBinary;var wasmModule;var ABORT=false;var EXITSTATUS;function assert(condition,text){if(!condition){abort(text)}}var isFileURI=filename=>filename.startsWith("file://");function growMemViews(){if(wasmMemory.buffer!=HEAP8.buffer){updateMemoryViews()}}if(ENVIRONMENT_IS_NODE&&ENVIRONMENT_IS_PTHREAD){var parentPort=worker_threads["parentPort"];parentPort.on("message",msg=>global.onmessage?.({data:msg}));Object.assign(globalThis,{self:global,postMessage:msg=>parentPort["postMessage"](msg)});process.on("uncaughtException",err=>{postMessage({cmd:"uncaughtException",error:err});process.exit(1)})}var startWorker;if(ENVIRONMENT_IS_PTHREAD){var initializedJS=false;self.onunhandledrejection=e=>{throw e.reason||e};async function handleMessage(e){try{var msgData=e["data"];var cmd=msgData.cmd;if(cmd==="load"){let messageQueue=[];self.onmessage=e=>messageQueue.push(e);startWorker=()=>{postMessage({cmd:"loaded"});for(let msg of messageQueue){handleMessage(msg)}self.onmessage=handleMessage};for(const handler of msgData.handlers){if(!Module[handler]||Module[handler].proxy){Module[handler]=(...args)=>{postMessage({cmd:"callHandler",handler,args})};if(handler=="print")out=Module[handler];if(handler=="printErr")err=Module[handler]}}wasmMemory=msgData.wasmMemory;updateMemoryViews();wasmModule=msgData.wasmModule;createWasm();run()}else if(cmd==="run"){establishStackSpace(msgData.pthread_ptr);__emscripten_thread_init(msgData.pthread_ptr,0,0,1,0,0);PThread.threadInitTLS();__emscripten_thread_mailbox_await(msgData.pthread_ptr);if(!initializedJS){initializedJS=true}try{await invokeEntryPoint(msgData.start_routine,msgData.arg)}catch(ex){if(ex!="unwind"){throw ex}}}else if(msgData.target==="setimmediate"){}else if(cmd==="checkMailbox"){if(initializedJS){checkMailbox()}}else if(cmd){err(`worker: received unknown command ${cmd}`);err(msgData)}}catch(ex){__emscripten_thread_crashed();throw ex}}self.onmessage=handleMessage}var HEAP8,HEAPU8,HEAP16,HEAPU16,HEAP32,HEAPU32,HEAPF32,HEAPF64;var HEAP64,HEAPU64;var runtimeInitialized=false;function updateMemoryViews(){var b=wasmMemory.buffer;HEAP8=new Int8Array(b);HEAP16=new Int16Array(b);Module["HEAPU8"]=HEAPU8=new Uint8Array(b);HEAPU16=new Uint16Array(b);HEAP32=new Int32Array(b);HEAPU32=new Uint32Array(b);HEAPF32=new Float32Array(b);HEAPF64=new Float64Array(b);HEAP64=new BigInt64Array(b);HEAPU64=new BigUint64Array(b)}function initMemory(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["wasmMemory"]){wasmMemory=Module["wasmMemory"]}else{var INITIAL_MEMORY=Module["INITIAL_MEMORY"]||134217728;wasmMemory=new WebAssembly.Memory({initial:BigInt(INITIAL_MEMORY/65536),maximum:65536n,shared:true,address:"i64"})}updateMemoryViews()}function preRun(){if(Module["preRun"]){if(typeof Module["preRun"]=="function")Module["preRun"]=[Module["preRun"]];while(Module["preRun"].length){addOnPreRun(Module["preRun"].shift())}}callRuntimeCallbacks(onPreRuns)}function initRuntime(){runtimeInitialized=true;if(ENVIRONMENT_IS_PTHREAD)return startWorker();if(!Module["noFSInit"]&&!FS.initialized)FS.init();TTY.init();wasmExports["__wasm_call_ctors"]();FS.ignorePermissions=false}function preMain(){}function postRun(){if(ENVIRONMENT_IS_PTHREAD){return}if(Module["postRun"]){if(typeof Module["postRun"]=="function")Module["postRun"]=[Module["postRun"]];while(Module["postRun"].length){addOnPostRun(Module["postRun"].shift())}}callRuntimeCallbacks(onPostRuns)}function abort(what){Module["onAbort"]?.(what);what="Aborted("+what+")";err(what);ABORT=true;what+=". Build with -sASSERTIONS for more info.";if(runtimeInitialized){___trap()}var e=new WebAssembly.RuntimeError(what);throw e}var wasmBinaryFile;function findWasmBinary(){return locateFile("wllama.wasm")}function getBinarySync(file){if(file==wasmBinaryFile&&wasmBinary){return new Uint8Array(wasmBinary)}if(readBinary){return readBinary(file)}throw"both async and sync fetching of the wasm failed"}async function getWasmBinary(binaryFile){if(!wasmBinary){try{var response=await readAsync(binaryFile);return new Uint8Array(response)}catch{}}return getBinarySync(binaryFile)}async function instantiateArrayBuffer(binaryFile,imports){try{var binary=await getWasmBinary(binaryFile);var instance=await WebAssembly.instantiate(binary,imports);return instance}catch(reason){err(`failed to asynchronously prepare wasm: ${reason}`);abort(reason)}}async function instantiateAsync(binary,binaryFile,imports){if(!binary&&!isFileURI(binaryFile)&&!ENVIRONMENT_IS_NODE){try{var response=fetch(binaryFile,{credentials:"same-origin"});var instantiationResult=await WebAssembly.instantiateStreaming(response,imports);return instantiationResult}catch(reason){err(`wasm streaming compile failed: ${reason}`);err("falling back to ArrayBuffer instantiation")}}return instantiateArrayBuffer(binaryFile,imports)}function getWasmImports(){assignWasmImports();if(!wasmImports.__instrumented){wasmImports.__instrumented=true;Asyncify.instrumentWasmImports(wasmImports)}var imports={env:wasmImports,wasi_snapshot_preview1:wasmImports};return imports}async function createWasm(){function receiveInstance(instance,module){wasmExports=instance.exports;wasmExports=Asyncify.instrumentWasmExports(wasmExports);wasmExports=applySignatureConversions(wasmExports);registerTLSInit(wasmExports["_emscripten_tls_init"]);assignWasmExports(wasmExports);wasmModule=module;removeRunDependency("wasm-instantiate");return wasmExports}addRunDependency("wasm-instantiate");function receiveInstantiationResult(result){return receiveInstance(result["instance"],result["module"])}var info=getWasmImports();if(Module["instantiateWasm"]){return new Promise((resolve,reject)=>{Module["instantiateWasm"](info,(inst,mod)=>{resolve(receiveInstance(inst,mod))})})}if(ENVIRONMENT_IS_PTHREAD){var instance=new WebAssembly.Instance(wasmModule,getWasmImports());return receiveInstance(instance,wasmModule)}wasmBinaryFile??=findWasmBinary();var result=await instantiateAsync(wasmBinary,wasmBinaryFile,info);var exports=receiveInstantiationResult(result);return exports}class ExitStatus{name="ExitStatus";constructor(status){this.message=`Program terminated with exit(${status})`;this.status=status}}var terminateWorker=worker=>{worker.terminate();worker.onmessage=e=>{}};var cleanupThread=pthread_ptr=>{var worker=PThread.pthreads[pthread_ptr];PThread.returnWorkerToPool(worker)};var callRuntimeCallbacks=callbacks=>{while(callbacks.length>0){callbacks.shift()(Module)}};var onPreRuns=[];var addOnPreRun=cb=>onPreRuns.push(cb);var runDependencies=0;var dependenciesFulfilled=null;var removeRunDependency=id=>{runDependencies--;Module["monitorRunDependencies"]?.(runDependencies);if(runDependencies==0){if(dependenciesFulfilled){var callback=dependenciesFulfilled;dependenciesFulfilled=null;callback()}}};var addRunDependency=id=>{runDependencies++;Module["monitorRunDependencies"]?.(runDependencies)};var spawnThread=threadParams=>{var worker=PThread.getNewWorker();if(!worker){return 6}PThread.runningWorkers.push(worker);PThread.pthreads[threadParams.pthread_ptr]=worker;worker.pthread_ptr=threadParams.pthread_ptr;var msg={cmd:"run",start_routine:threadParams.startRoutine,arg:threadParams.arg,pthread_ptr:threadParams.pthread_ptr};if(ENVIRONMENT_IS_NODE){worker.unref()}worker.postMessage(msg,threadParams.transferList);return 0};var runtimeKeepaliveCounter=0;var keepRuntimeAlive=()=>noExitRuntime||runtimeKeepaliveCounter>0;var stackSave=()=>_emscripten_stack_get_current();var stackRestore=val=>__emscripten_stack_restore(val);var stackAlloc=sz=>__emscripten_stack_alloc(sz);var proxyToMainThread=(funcIndex,emAsmAddr,sync,...callArgs)=>{var serializedNumCallArgs=callArgs.length*2;var sp=stackSave();var args=stackAlloc(serializedNumCallArgs*8);var b=args/8;for(var i=0;i<callArgs.length;i++){var arg=callArgs[i];if(typeof arg=="bigint"){(growMemViews(),HEAP64)[b+2*i]=1n;(growMemViews(),HEAP64)[b+2*i+1]=arg}else{(growMemViews(),HEAP64)[b+2*i]=0n;(growMemViews(),HEAPF64)[b+2*i+1]=arg}}var rtn=__emscripten_run_js_on_main_thread(funcIndex,emAsmAddr,serializedNumCallArgs,args,sync);stackRestore(sp);return rtn};function _proc_exit(code){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(0,0,1,code);EXITSTATUS=code;if(!keepRuntimeAlive()){PThread.terminateAllThreads();Module["onExit"]?.(code);ABORT=true}quit_(code,new ExitStatus(code))}function exitOnMainThread(returnCode){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(1,0,0,returnCode);_exit(returnCode)}var exitJS=(status,implicit)=>{EXITSTATUS=status;if(ENVIRONMENT_IS_PTHREAD){exitOnMainThread(status);throw"unwind"}_proc_exit(status)};var _exit=exitJS;var PThread={unusedWorkers:[],runningWorkers:[],tlsInitFunctions:[],pthreads:{},init(){if(!ENVIRONMENT_IS_PTHREAD){PThread.initMainThread()}},initMainThread(){var pthreadPoolSize=Module["pthreadPoolSize"];while(pthreadPoolSize--){PThread.allocateUnusedWorker()}addOnPreRun(async()=>{var pthreadPoolReady=PThread.loadWasmModuleToAllWorkers();addRunDependency("loading-workers");await pthreadPoolReady;removeRunDependency("loading-workers")})},terminateAllThreads:()=>{for(var worker of PThread.runningWorkers){terminateWorker(worker)}for(var worker of PThread.unusedWorkers){terminateWorker(worker)}PThread.unusedWorkers=[];PThread.runningWorkers=[];PThread.pthreads={}},returnWorkerToPool:worker=>{var pthread_ptr=worker.pthread_ptr;delete PThread.pthreads[pthread_ptr];PThread.unusedWorkers.push(worker);PThread.runningWorkers.splice(PThread.runningWorkers.indexOf(worker),1);worker.pthread_ptr=0;__emscripten_thread_free_data(pthread_ptr)},threadInitTLS(){PThread.tlsInitFunctions.forEach(f=>f())},loadWasmModuleToWorker:worker=>new Promise(onFinishedLoading=>{worker.onmessage=e=>{var d=e["data"];var cmd=d.cmd;if(d.targetThread&&d.targetThread!=_pthread_self()){var targetWorker=PThread.pthreads[d.targetThread];if(targetWorker){targetWorker.postMessage(d,d.transferList)}else{err(`Internal error! Worker sent a message "${cmd}" to target pthread ${d.targetThread}, but that thread no longer exists!`)}return}if(cmd==="checkMailbox"){checkMailbox()}else if(cmd==="spawnThread"){spawnThread(d)}else if(cmd==="cleanupThread"){callUserCallback(()=>cleanupThread(d.thread))}else if(cmd==="loaded"){worker.loaded=true;if(ENVIRONMENT_IS_NODE&&!worker.pthread_ptr){worker.unref()}onFinishedLoading(worker)}else if(d.target==="setimmediate"){worker.postMessage(d)}else if(cmd==="uncaughtException"){worker.onerror(d.error)}else if(cmd==="callHandler"){Module[d.handler](...d.args)}else if(cmd){err(`worker sent an unknown command ${cmd}`)}};worker.onerror=e=>{var message="worker sent an error!";err(`${message} ${e.filename}:${e.lineno}: ${e.message}`);throw e};if(ENVIRONMENT_IS_NODE){worker.on("message",data=>worker.onmessage({data}));worker.on("error",e=>worker.onerror(e))}var handlers=[];var knownHandlers=["onExit","onAbort","print","printErr"];for(var handler of knownHandlers){if(Module.propertyIsEnumerable(handler)){handlers.push(handler)}}worker.postMessage({cmd:"load",handlers,wasmMemory,wasmModule})}),async loadWasmModuleToAllWorkers(){if(ENVIRONMENT_IS_PTHREAD){return}let pthreadPoolReady=Promise.all(PThread.unusedWorkers.map(PThread.loadWasmModuleToWorker));return pthreadPoolReady},allocateUnusedWorker(){var worker;var pthreadMainJs=_scriptName;if(Module["mainScriptUrlOrBlob"]){pthreadMainJs=Module["mainScriptUrlOrBlob"];if(typeof pthreadMainJs!="string"){pthreadMainJs=URL.createObjectURL(pthreadMainJs)}}worker=new Worker(pthreadMainJs,{workerData:"em-pthread",name:"em-pthread"});PThread.unusedWorkers.push(worker)},getNewWorker(){if(PThread.unusedWorkers.length==0){PThread.allocateUnusedWorker();PThread.loadWasmModuleToWorker(PThread.unusedWorkers[0])}return PThread.unusedWorkers.pop()}};var onPostRuns=[];var addOnPostRun=cb=>onPostRuns.push(cb);function establishStackSpace(pthread_ptr){var stackHigh=Number((growMemViews(),HEAPU64)[(pthread_ptr+88)/8]);var stackSize=Number((growMemViews(),HEAPU64)[(pthread_ptr+96)/8]);var stackLow=stackHigh-stackSize;_emscripten_stack_set_limits(stackHigh,stackLow);stackRestore(stackHigh)}var wasmTableMirror=[];var getWasmTableEntry=funcPtr=>{funcPtr=Number(funcPtr);var func=wasmTableMirror[funcPtr];if(!func){wasmTableMirror[funcPtr]=func=wasmTable.get(BigInt(funcPtr));if(Asyncify.isAsyncExport(func)){wasmTableMirror[funcPtr]=func=Asyncify.makeAsyncFunction(func)}}return func};var invokeEntryPoint=async(ptr,arg)=>{runtimeKeepaliveCounter=0;noExitRuntime=0;var result=(a1=>WebAssembly.promising(getWasmTableEntry(ptr)).call(null,BigInt(a1)))(arg);function finish(result){if(keepRuntimeAlive()){EXITSTATUS=result;return}__emscripten_thread_exit(result)}result=await result;finish(result)};invokeEntryPoint.isAsync=true;var noExitRuntime=true;var registerTLSInit=tlsInitFunc=>PThread.tlsInitFunctions.push(tlsInitFunc);var wasmMemory;function pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(2,0,1,pthread_ptr,attr,startRoutine,arg);return ___pthread_create_js(pthread_ptr,attr,startRoutine,arg)}var _emscripten_has_threading_support=()=>!!globalThis.SharedArrayBuffer;var INT53_MAX=9007199254740992;var INT53_MIN=-9007199254740992;var bigintToI53Checked=num=>num<INT53_MIN||num>INT53_MAX?NaN:Number(num);function ___pthread_create_js(pthread_ptr,attr,startRoutine,arg){pthread_ptr=bigintToI53Checked(pthread_ptr);attr=bigintToI53Checked(attr);startRoutine=bigintToI53Checked(startRoutine);arg=bigintToI53Checked(arg);if(!_emscripten_has_threading_support()){return 6}var transferList=[];var error=0;if(ENVIRONMENT_IS_PTHREAD&&(transferList.length===0||error)){return pthreadCreateProxied(pthread_ptr,attr,startRoutine,arg)}if(error)return error;var threadParams={startRoutine,pthread_ptr,arg,transferList};if(ENVIRONMENT_IS_PTHREAD){threadParams.cmd="spawnThread";postMessage(threadParams,transferList);return 0}return spawnThread(threadParams)}var syscallGetVarargP=()=>{var ret=Number((growMemViews(),HEAPU64)[SYSCALLS.varargs/8]);SYSCALLS.varargs+=8;return ret};var syscallGetVarargI=()=>{var ret=(growMemViews(),HEAP32)[+SYSCALLS.varargs/4];SYSCALLS.varargs+=4;return ret};var PATH={isAbs:path=>path.charAt(0)==="/",splitPath:filename=>{var splitPathRe=/^(\\/?|)([\\s\\S]*?)((?:\\.{1,2}|[^\\/]+?|)(\\.[^.\\/]*|))(?:[\\/]*)$/;return splitPathRe.exec(filename).slice(1)},normalizeArray:(parts,allowAboveRoot)=>{var up=0;for(var i=parts.length-1;i>=0;i--){var last=parts[i];if(last==="."){parts.splice(i,1)}else if(last===".."){parts.splice(i,1);up++}else if(up){parts.splice(i,1);up--}}if(allowAboveRoot){for(;up;up--){parts.unshift("..")}}return parts},normalize:path=>{var isAbsolute=PATH.isAbs(path),trailingSlash=path.slice(-1)==="/";path=PATH.normalizeArray(path.split("/").filter(p=>!!p),!isAbsolute).join("/");if(!path&&!isAbsolute){path="."}if(path&&trailingSlash){path+="/"}return(isAbsolute?"/":"")+path},dirname:path=>{var result=PATH.splitPath(path),root=result[0],dir=result[1];if(!root&&!dir){return"."}if(dir){dir=dir.slice(0,-1)}return root+dir},basename:path=>path&&path.match(/([^\\/]+|\\/)\\/*$/)[1],join:(...paths)=>PATH.normalize(paths.join("/")),join2:(l,r)=>PATH.normalize(l+"/"+r)};var initRandomFill=()=>view=>view.set(crypto.getRandomValues(new Uint8Array(view.byteLength)));var randomFill=view=>{(randomFill=initRandomFill())(view)};var PATH_FS={resolve:(...args)=>{var resolvedPath="",resolvedAbsolute=false;for(var i=args.length-1;i>=-1&&!resolvedAbsolute;i--){var path=i>=0?args[i]:FS.cwd();if(typeof path!="string"){throw new TypeError("Arguments to path.resolve must be strings")}else if(!path){return""}resolvedPath=path+"/"+resolvedPath;resolvedAbsolute=PATH.isAbs(path)}resolvedPath=PATH.normalizeArray(resolvedPath.split("/").filter(p=>!!p),!resolvedAbsolute).join("/");return(resolvedAbsolute?"/":"")+resolvedPath||"."},relative:(from,to)=>{from=PATH_FS.resolve(from).slice(1);to=PATH_FS.resolve(to).slice(1);function trim(arr){var start=0;for(;start<arr.length;start++){if(arr[start]!=="")break}var end=arr.length-1;for(;end>=0;end--){if(arr[end]!=="")break}if(start>end)return[];return arr.slice(start,end-start+1)}var fromParts=trim(from.split("/"));var toParts=trim(to.split("/"));var length=Math.min(fromParts.length,toParts.length);var samePartsLength=length;for(var i=0;i<length;i++){if(fromParts[i]!==toParts[i]){samePartsLength=i;break}}var outputParts=[];for(var i=samePartsLength;i<fromParts.length;i++){outputParts.push("..")}outputParts=outputParts.concat(toParts.slice(samePartsLength));return outputParts.join("/")}};var UTF8Decoder=globalThis.TextDecoder&&new TextDecoder;var findStringEnd=(heapOrArray,idx,maxBytesToRead,ignoreNul)=>{var maxIdx=idx+maxBytesToRead;if(ignoreNul)return maxIdx;while(heapOrArray[idx]&&!(idx>=maxIdx))++idx;return idx};var UTF8ArrayToString=(heapOrArray,idx=0,maxBytesToRead,ignoreNul)=>{var endPtr=findStringEnd(heapOrArray,idx,maxBytesToRead,ignoreNul);if(endPtr-idx>16&&heapOrArray.buffer&&UTF8Decoder){return UTF8Decoder.decode(heapOrArray.buffer instanceof ArrayBuffer?heapOrArray.subarray(idx,endPtr):heapOrArray.slice(idx,endPtr))}var str="";while(idx<endPtr){var u0=heapOrArray[idx++];if(!(u0&128)){str+=String.fromCharCode(u0);continue}var u1=heapOrArray[idx++]&63;if((u0&224)==192){str+=String.fromCharCode((u0&31)<<6|u1);continue}var u2=heapOrArray[idx++]&63;if((u0&240)==224){u0=(u0&15)<<12|u1<<6|u2}else{u0=(u0&7)<<18|u1<<12|u2<<6|heapOrArray[idx++]&63}if(u0<65536){str+=String.fromCharCode(u0)}else{var ch=u0-65536;str+=String.fromCharCode(55296|ch>>10,56320|ch&1023)}}return str};var FS_stdin_getChar_buffer=[];var lengthBytesUTF8=str=>{var len=0;for(var i=0;i<str.length;++i){var c=str.charCodeAt(i);if(c<=127){len++}else if(c<=2047){len+=2}else if(c>=55296&&c<=57343){len+=4;++i}else{len+=3}}return len};var stringToUTF8Array=(str,heap,outIdx,maxBytesToWrite)=>{if(!(maxBytesToWrite>0))return 0;var startIdx=outIdx;var endIdx=outIdx+maxBytesToWrite-1;for(var i=0;i<str.length;++i){var u=str.codePointAt(i);if(u<=127){if(outIdx>=endIdx)break;heap[outIdx++]=u}else if(u<=2047){if(outIdx+1>=endIdx)break;heap[outIdx++]=192|u>>6;heap[outIdx++]=128|u&63}else if(u<=65535){if(outIdx+2>=endIdx)break;heap[outIdx++]=224|u>>12;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63}else{if(outIdx+3>=endIdx)break;heap[outIdx++]=240|u>>18;heap[outIdx++]=128|u>>12&63;heap[outIdx++]=128|u>>6&63;heap[outIdx++]=128|u&63;i++}}heap[outIdx]=0;return outIdx-startIdx};var intArrayFromString=(stringy,dontAddNull,length)=>{var len=length>0?length:lengthBytesUTF8(stringy)+1;var u8array=new Array(len);var numBytesWritten=stringToUTF8Array(stringy,u8array,0,u8array.length);if(dontAddNull)u8array.length=numBytesWritten;return u8array};var FS_stdin_getChar=()=>{if(!FS_stdin_getChar_buffer.length){var result=null;if(ENVIRONMENT_IS_NODE){var BUFSIZE=256;var buf=Buffer.alloc(BUFSIZE);var bytesRead=0;var fd=process.stdin.fd;try{bytesRead=fs.readSync(fd,buf,0,BUFSIZE)}catch(e){if(e.toString().includes("EOF"))bytesRead=0;else throw e}if(bytesRead>0){result=buf.slice(0,bytesRead).toString("utf-8")}}else if(globalThis.window?.prompt){result=window.prompt("Input: ");if(result!==null){result+="\\n"}}else{}if(!result){return null}FS_stdin_getChar_buffer=intArrayFromString(result,true)}return FS_stdin_getChar_buffer.shift()};var TTY={ttys:[],init(){},shutdown(){},register(dev,ops){TTY.ttys[dev]={input:[],output:[],ops};FS.registerDevice(dev,TTY.stream_ops)},stream_ops:{open(stream){var tty=TTY.ttys[stream.node.rdev];if(!tty){throw new FS.ErrnoError(43)}stream.tty=tty;stream.seekable=false},close(stream){stream.tty.ops.fsync(stream.tty)},fsync(stream){stream.tty.ops.fsync(stream.tty)},read(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.get_char){throw new FS.ErrnoError(60)}var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=stream.tty.ops.get_char(stream.tty)}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){if(!stream.tty||!stream.tty.ops.put_char){throw new FS.ErrnoError(60)}try{for(var i=0;i<length;i++){stream.tty.ops.put_char(stream.tty,buffer[offset+i])}}catch(e){throw new FS.ErrnoError(29)}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}},default_tty_ops:{get_char(tty){return FS_stdin_getChar()},put_char(tty,val){if(val===null||val===10){out(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){out(UTF8ArrayToString(tty.output));tty.output=[]}},ioctl_tcgets(tty){return{c_iflag:25856,c_oflag:5,c_cflag:191,c_lflag:35387,c_cc:[3,28,127,21,4,0,1,0,17,19,26,0,18,15,23,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}},ioctl_tcsets(tty,optional_actions,data){return 0},ioctl_tiocgwinsz(tty){return[24,80]}},default_tty1_ops:{put_char(tty,val){if(val===null||val===10){err(UTF8ArrayToString(tty.output));tty.output=[]}else{if(val!=0)tty.output.push(val)}},fsync(tty){if(tty.output?.length>0){err(UTF8ArrayToString(tty.output));tty.output=[]}}}};var zeroMemory=(ptr,size)=>(growMemViews(),HEAPU8).fill(0,ptr,ptr+size);var alignMemory=(size,alignment)=>Math.ceil(size/alignment)*alignment;var mmapAlloc=size=>{size=alignMemory(size,65536);var ptr=_emscripten_builtin_memalign(65536,size);if(ptr)zeroMemory(ptr,size);return ptr};var MEMFS={ops_table:null,mount(mount){return MEMFS.createNode(null,"/",16895,0)},createNode(parent,name,mode,dev){if(FS.isBlkdev(mode)||FS.isFIFO(mode)){throw new FS.ErrnoError(63)}MEMFS.ops_table||={dir:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,lookup:MEMFS.node_ops.lookup,mknod:MEMFS.node_ops.mknod,rename:MEMFS.node_ops.rename,unlink:MEMFS.node_ops.unlink,rmdir:MEMFS.node_ops.rmdir,readdir:MEMFS.node_ops.readdir,symlink:MEMFS.node_ops.symlink},stream:{llseek:MEMFS.stream_ops.llseek}},file:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:{llseek:MEMFS.stream_ops.llseek,read:MEMFS.stream_ops.read,write:MEMFS.stream_ops.write,mmap:MEMFS.stream_ops.mmap,msync:MEMFS.stream_ops.msync}},link:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr,readlink:MEMFS.node_ops.readlink},stream:{}},chrdev:{node:{getattr:MEMFS.node_ops.getattr,setattr:MEMFS.node_ops.setattr},stream:FS.chrdev_stream_ops}};var node=FS.createNode(parent,name,mode,dev);if(FS.isDir(node.mode)){node.node_ops=MEMFS.ops_table.dir.node;node.stream_ops=MEMFS.ops_table.dir.stream;node.contents={}}else if(FS.isFile(node.mode)){node.node_ops=MEMFS.ops_table.file.node;node.stream_ops=MEMFS.ops_table.file.stream;node.usedBytes=0;node.contents=null}else if(FS.isLink(node.mode)){node.node_ops=MEMFS.ops_table.link.node;node.stream_ops=MEMFS.ops_table.link.stream}else if(FS.isChrdev(node.mode)){node.node_ops=MEMFS.ops_table.chrdev.node;node.stream_ops=MEMFS.ops_table.chrdev.stream}node.atime=node.mtime=node.ctime=Date.now();if(parent){parent.contents[name]=node;parent.atime=parent.mtime=parent.ctime=node.atime}return node},getFileDataAsTypedArray(node){if(!node.contents)return new Uint8Array(0);if(node.contents.subarray)return node.contents.subarray(0,node.usedBytes);return new Uint8Array(node.contents)},expandFileStorage(node,newCapacity){var prevCapacity=node.contents?node.contents.length:0;if(prevCapacity>=newCapacity)return;var CAPACITY_DOUBLING_MAX=1024*1024;newCapacity=Math.max(newCapacity,prevCapacity*(prevCapacity<CAPACITY_DOUBLING_MAX?2:1.125)>>>0);if(prevCapacity!=0)newCapacity=Math.max(newCapacity,256);var oldContents=node.contents;node.contents=new Uint8Array(newCapacity);if(node.usedBytes>0)node.contents.set(oldContents.subarray(0,node.usedBytes),0)},resizeFileStorage(node,newSize){if(node.usedBytes==newSize)return;if(newSize==0){node.contents=null;node.usedBytes=0}else{var oldContents=node.contents;node.contents=new Uint8Array(newSize);if(oldContents){node.contents.set(oldContents.subarray(0,Math.min(newSize,node.usedBytes)))}node.usedBytes=newSize}},node_ops:{getattr(node){var attr={};attr.dev=FS.isChrdev(node.mode)?node.id:1;attr.ino=node.id;attr.mode=node.mode;attr.nlink=1;attr.uid=0;attr.gid=0;attr.rdev=node.rdev;if(FS.isDir(node.mode)){attr.size=4096}else if(FS.isFile(node.mode)){attr.size=node.usedBytes}else if(FS.isLink(node.mode)){attr.size=node.link.length}else{attr.size=0}attr.atime=new Date(node.atime);attr.mtime=new Date(node.mtime);attr.ctime=new Date(node.ctime);attr.blksize=4096;attr.blocks=Math.ceil(attr.size/attr.blksize);return attr},setattr(node,attr){for(const key of["mode","atime","mtime","ctime"]){if(attr[key]!=null){node[key]=attr[key]}}if(attr.size!==undefined){MEMFS.resizeFileStorage(node,attr.size)}},lookup(parent,name){if(!MEMFS.doesNotExistError){MEMFS.doesNotExistError=new FS.ErrnoError(44);MEMFS.doesNotExistError.stack="<generic error, no stack>"}throw MEMFS.doesNotExistError},mknod(parent,name,mode,dev){return MEMFS.createNode(parent,name,mode,dev)},rename(old_node,new_dir,new_name){var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(new_node){if(FS.isDir(old_node.mode)){for(var i in new_node.contents){throw new FS.ErrnoError(55)}}FS.hashRemoveNode(new_node)}delete old_node.parent.contents[old_node.name];new_dir.contents[new_name]=old_node;old_node.name=new_name;new_dir.ctime=new_dir.mtime=old_node.parent.ctime=old_node.parent.mtime=Date.now()},unlink(parent,name){delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},rmdir(parent,name){var node=FS.lookupNode(parent,name);for(var i in node.contents){throw new FS.ErrnoError(55)}delete parent.contents[name];parent.ctime=parent.mtime=Date.now()},readdir(node){return[".","..",...Object.keys(node.contents)]},symlink(parent,newname,oldpath){var node=MEMFS.createNode(parent,newname,511|40960,0);node.link=oldpath;return node},readlink(node){if(!FS.isLink(node.mode)){throw new FS.ErrnoError(28)}return node.link}},stream_ops:{read(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=stream.node.usedBytes)return 0;var size=Math.min(stream.node.usedBytes-position,length);if(size>8&&contents.subarray){buffer.set(contents.subarray(position,position+size),offset)}else{for(var i=0;i<size;i++)buffer[offset+i]=contents[position+i]}return size},write(stream,buffer,offset,length,position,canOwn){if(buffer.buffer===(growMemViews(),HEAP8).buffer){canOwn=false}if(!length)return 0;var node=stream.node;node.mtime=node.ctime=Date.now();if(buffer.subarray&&(!node.contents||node.contents.subarray)){if(canOwn){node.contents=buffer.subarray(offset,offset+length);node.usedBytes=length;return length}else if(node.usedBytes===0&&position===0){node.contents=buffer.slice(offset,offset+length);node.usedBytes=length;return length}else if(position+length<=node.usedBytes){node.contents.set(buffer.subarray(offset,offset+length),position);return length}}MEMFS.expandFileStorage(node,position+length);if(node.contents.subarray&&buffer.subarray){node.contents.set(buffer.subarray(offset,offset+length),position)}else{for(var i=0;i<length;i++){node.contents[position+i]=buffer[offset+i]}}node.usedBytes=Math.max(node.usedBytes,position+length);return length},llseek(stream,offset,whence){var position=offset;if(whence===1){position+=stream.position}else if(whence===2){if(FS.isFile(stream.node.mode)){position+=stream.node.usedBytes}}if(position<0){throw new FS.ErrnoError(28)}return position},mmap(stream,length,position,prot,flags){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}var ptr;var allocated;var contents=stream.node.contents;if(!(flags&2)&&contents&&contents.buffer===(growMemViews(),HEAP8).buffer){allocated=false;ptr=contents.byteOffset}else{allocated=true;ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}if(contents){if(position>0||position+length<contents.length){if(contents.subarray){contents=contents.subarray(position,position+length)}else{contents=Array.prototype.slice.call(contents,position,position+length)}}(growMemViews(),HEAP8).set(contents,ptr)}}return{ptr,allocated}},msync(stream,buffer,offset,length,mmapFlags){MEMFS.stream_ops.write(stream,buffer,0,length,offset,false);return 0}}};var FS_modeStringToFlags=str=>{var flagModes={r:0,"r+":2,w:512|64|1,"w+":512|64|2,a:1024|64|1,"a+":1024|64|2};var flags=flagModes[str];if(typeof flags=="undefined"){throw new Error(`Unknown file open mode: ${str}`)}return flags};var FS_getMode=(canRead,canWrite)=>{var mode=0;if(canRead)mode|=292|73;if(canWrite)mode|=146;return mode};var asyncLoad=async url=>{var arrayBuffer=await readAsync(url);return new Uint8Array(arrayBuffer)};var FS_createDataFile=(...args)=>FS.createDataFile(...args);var getUniqueRunDependency=id=>id;var preloadPlugins=[];var FS_handledByPreloadPlugin=async(byteArray,fullname)=>{if(typeof Browser!="undefined")Browser.init();for(var plugin of preloadPlugins){if(plugin["canHandle"](fullname)){return plugin["handle"](byteArray,fullname)}}return byteArray};var FS_preloadFile=async(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish)=>{var fullname=name?PATH_FS.resolve(PATH.join2(parent,name)):parent;var dep=getUniqueRunDependency(`cp ${fullname}`);addRunDependency(dep);try{var byteArray=url;if(typeof url=="string"){byteArray=await asyncLoad(url)}byteArray=await FS_handledByPreloadPlugin(byteArray,fullname);preFinish?.();if(!dontCreateFile){FS_createDataFile(parent,name,byteArray,canRead,canWrite,canOwn)}}finally{removeRunDependency(dep)}};var FS_createPreloadedFile=(parent,name,url,canRead,canWrite,onload,onerror,dontCreateFile,canOwn,preFinish)=>{FS_preloadFile(parent,name,url,canRead,canWrite,dontCreateFile,canOwn,preFinish).then(onload).catch(onerror)};var FS={root:null,mounts:[],devices:{},streams:[],nextInode:1,nameTable:null,currentPath:"/",initialized:false,ignorePermissions:true,filesystems:null,syncFSRequests:0,readFiles:{},ErrnoError:class{name="ErrnoError";constructor(errno){this.errno=errno}},FSStream:class{shared={};get object(){return this.node}set object(val){this.node=val}get isRead(){return(this.flags&2097155)!==1}get isWrite(){return(this.flags&2097155)!==0}get isAppend(){return this.flags&1024}get flags(){return this.shared.flags}set flags(val){this.shared.flags=val}get position(){return this.shared.position}set position(val){this.shared.position=val}},FSNode:class{node_ops={};stream_ops={};readMode=292|73;writeMode=146;mounted=null;constructor(parent,name,mode,rdev){if(!parent){parent=this}this.parent=parent;this.mount=parent.mount;this.id=FS.nextInode++;this.name=name;this.mode=mode;this.rdev=rdev;this.atime=this.mtime=this.ctime=Date.now()}get read(){return(this.mode&this.readMode)===this.readMode}set read(val){val?this.mode|=this.readMode:this.mode&=~this.readMode}get write(){return(this.mode&this.writeMode)===this.writeMode}set write(val){val?this.mode|=this.writeMode:this.mode&=~this.writeMode}get isFolder(){return FS.isDir(this.mode)}get isDevice(){return FS.isChrdev(this.mode)}},lookupPath(path,opts={}){if(!path){throw new FS.ErrnoError(44)}opts.follow_mount??=true;if(!PATH.isAbs(path)){path=FS.cwd()+"/"+path}linkloop:for(var nlinks=0;nlinks<40;nlinks++){var parts=path.split("/").filter(p=>!!p);var current=FS.root;var current_path="/";for(var i=0;i<parts.length;i++){var islast=i===parts.length-1;if(islast&&opts.parent){break}if(parts[i]==="."){continue}if(parts[i]===".."){current_path=PATH.dirname(current_path);if(FS.isRoot(current)){path=current_path+"/"+parts.slice(i+1).join("/");nlinks--;continue linkloop}else{current=current.parent}continue}current_path=PATH.join2(current_path,parts[i]);try{current=FS.lookupNode(current,parts[i])}catch(e){if(e?.errno===44&&islast&&opts.noent_okay){return{path:current_path}}throw e}if(FS.isMountpoint(current)&&(!islast||opts.follow_mount)){current=current.mounted.root}if(FS.isLink(current.mode)&&(!islast||opts.follow)){if(!current.node_ops.readlink){throw new FS.ErrnoError(52)}var link=current.node_ops.readlink(current);if(!PATH.isAbs(link)){link=PATH.dirname(current_path)+"/"+link}path=link+"/"+parts.slice(i+1).join("/");continue linkloop}}return{path:current_path,node:current}}throw new FS.ErrnoError(32)},getPath(node){var path;while(true){if(FS.isRoot(node)){var mount=node.mount.mountpoint;if(!path)return mount;return mount[mount.length-1]!=="/"?`${mount}/${path}`:mount+path}path=path?`${node.name}/${path}`:node.name;node=node.parent}},hashName(parentid,name){var hash=0;for(var i=0;i<name.length;i++){hash=(hash<<5)-hash+name.charCodeAt(i)|0}return(parentid+hash>>>0)%FS.nameTable.length},hashAddNode(node){var hash=FS.hashName(node.parent.id,node.name);node.name_next=FS.nameTable[hash];FS.nameTable[hash]=node},hashRemoveNode(node){var hash=FS.hashName(node.parent.id,node.name);if(FS.nameTable[hash]===node){FS.nameTable[hash]=node.name_next}else{var current=FS.nameTable[hash];while(current){if(current.name_next===node){current.name_next=node.name_next;break}current=current.name_next}}},lookupNode(parent,name){var errCode=FS.mayLookup(parent);if(errCode){throw new FS.ErrnoError(errCode)}var hash=FS.hashName(parent.id,name);for(var node=FS.nameTable[hash];node;node=node.name_next){var nodeName=node.name;if(node.parent.id===parent.id&&nodeName===name){return node}}return FS.lookup(parent,name)},createNode(parent,name,mode,rdev){var node=new FS.FSNode(parent,name,mode,rdev);FS.hashAddNode(node);return node},destroyNode(node){FS.hashRemoveNode(node)},isRoot(node){return node===node.parent},isMountpoint(node){return!!node.mounted},isFile(mode){return(mode&61440)===32768},isDir(mode){return(mode&61440)===16384},isLink(mode){return(mode&61440)===40960},isChrdev(mode){return(mode&61440)===8192},isBlkdev(mode){return(mode&61440)===24576},isFIFO(mode){return(mode&61440)===4096},isSocket(mode){return(mode&49152)===49152},flagsToPermissionString(flag){var perms=["r","w","rw"][flag&3];if(flag&512){perms+="w"}return perms},nodePermissions(node,perms){if(FS.ignorePermissions){return 0}if(perms.includes("r")&&!(node.mode&292)){return 2}else if(perms.includes("w")&&!(node.mode&146)){return 2}else if(perms.includes("x")&&!(node.mode&73)){return 2}return 0},mayLookup(dir){if(!FS.isDir(dir.mode))return 54;var errCode=FS.nodePermissions(dir,"x");if(errCode)return errCode;if(!dir.node_ops.lookup)return 2;return 0},mayCreate(dir,name){if(!FS.isDir(dir.mode)){return 54}try{var node=FS.lookupNode(dir,name);return 20}catch(e){}return FS.nodePermissions(dir,"wx")},mayDelete(dir,name,isdir){var node;try{node=FS.lookupNode(dir,name)}catch(e){return e.errno}var errCode=FS.nodePermissions(dir,"wx");if(errCode){return errCode}if(isdir){if(!FS.isDir(node.mode)){return 54}if(FS.isRoot(node)||FS.getPath(node)===FS.cwd()){return 10}}else{if(FS.isDir(node.mode)){return 31}}return 0},mayOpen(node,flags){if(!node){return 44}if(FS.isLink(node.mode)){return 32}else if(FS.isDir(node.mode)){if(FS.flagsToPermissionString(flags)!=="r"||flags&(512|64)){return 31}}return FS.nodePermissions(node,FS.flagsToPermissionString(flags))},checkOpExists(op,err){if(!op){throw new FS.ErrnoError(err)}return op},MAX_OPEN_FDS:4096,nextfd(){for(var fd=0;fd<=FS.MAX_OPEN_FDS;fd++){if(!FS.streams[fd]){return fd}}throw new FS.ErrnoError(33)},getStreamChecked(fd){var stream=FS.getStream(fd);if(!stream){throw new FS.ErrnoError(8)}return stream},getStream:fd=>FS.streams[fd],createStream(stream,fd=-1){stream=Object.assign(new FS.FSStream,stream);if(fd==-1){fd=FS.nextfd()}stream.fd=fd;FS.streams[fd]=stream;return stream},closeStream(fd){FS.streams[fd]=null},dupStream(origStream,fd=-1){var stream=FS.createStream(origStream,fd);stream.stream_ops?.dup?.(stream);return stream},doSetAttr(stream,node,attr){var setattr=stream?.stream_ops.setattr;var arg=setattr?stream:node;setattr??=node.node_ops.setattr;FS.checkOpExists(setattr,63);setattr(arg,attr)},chrdev_stream_ops:{open(stream){var device=FS.getDevice(stream.node.rdev);stream.stream_ops=device.stream_ops;stream.stream_ops.open?.(stream)},llseek(){throw new FS.ErrnoError(70)}},major:dev=>dev>>8,minor:dev=>dev&255,makedev:(ma,mi)=>ma<<8|mi,registerDevice(dev,ops){FS.devices[dev]={stream_ops:ops}},getDevice:dev=>FS.devices[dev],getMounts(mount){var mounts=[];var check=[mount];while(check.length){var m=check.pop();mounts.push(m);check.push(...m.mounts)}return mounts},syncfs(populate,callback){if(typeof populate=="function"){callback=populate;populate=false}FS.syncFSRequests++;if(FS.syncFSRequests>1){err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`)}var mounts=FS.getMounts(FS.root.mount);var completed=0;function doCallback(errCode){FS.syncFSRequests--;return callback(errCode)}function done(errCode){if(errCode){if(!done.errored){done.errored=true;return doCallback(errCode)}return}if(++completed>=mounts.length){doCallback(null)}}for(var mount of mounts){if(mount.type.syncfs){mount.type.syncfs(mount,populate,done)}else{done(null)}}},mount(type,opts,mountpoint){var root=mountpoint==="/";var pseudo=!mountpoint;var node;if(root&&FS.root){throw new FS.ErrnoError(10)}else if(!root&&!pseudo){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});mountpoint=lookup.path;node=lookup.node;if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}if(!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}}var mount={type,opts,mountpoint,mounts:[]};var mountRoot=type.mount(mount);mountRoot.mount=mount;mount.root=mountRoot;if(root){FS.root=mountRoot}else if(node){node.mounted=mount;if(node.mount){node.mount.mounts.push(mount)}}return mountRoot},unmount(mountpoint){var lookup=FS.lookupPath(mountpoint,{follow_mount:false});if(!FS.isMountpoint(lookup.node)){throw new FS.ErrnoError(28)}var node=lookup.node;var mount=node.mounted;var mounts=FS.getMounts(mount);for(var[hash,current]of Object.entries(FS.nameTable)){while(current){var next=current.name_next;if(mounts.includes(current.mount)){FS.destroyNode(current)}current=next}}node.mounted=null;var idx=node.mount.mounts.indexOf(mount);node.mount.mounts.splice(idx,1)},lookup(parent,name){return parent.node_ops.lookup(parent,name)},mknod(path,mode,dev){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);if(!name){throw new FS.ErrnoError(28)}if(name==="."||name===".."){throw new FS.ErrnoError(20)}var errCode=FS.mayCreate(parent,name);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.mknod){throw new FS.ErrnoError(63)}return parent.node_ops.mknod(parent,name,mode,dev)},statfs(path){return FS.statfsNode(FS.lookupPath(path,{follow:true}).node)},statfsStream(stream){return FS.statfsNode(stream.node)},statfsNode(node){var rtn={bsize:4096,frsize:4096,blocks:1e6,bfree:5e5,bavail:5e5,files:FS.nextInode,ffree:FS.nextInode-1,fsid:42,flags:2,namelen:255};if(node.node_ops.statfs){Object.assign(rtn,node.node_ops.statfs(node.mount.opts.root))}return rtn},create(path,mode=438){mode&=4095;mode|=32768;return FS.mknod(path,mode,0)},mkdir(path,mode=511){mode&=511|512;mode|=16384;return FS.mknod(path,mode,0)},mkdirTree(path,mode){var dirs=path.split("/");var d="";for(var dir of dirs){if(!dir)continue;if(d||PATH.isAbs(path))d+="/";d+=dir;try{FS.mkdir(d,mode)}catch(e){if(e.errno!=20)throw e}}},mkdev(path,mode,dev){if(typeof dev=="undefined"){dev=mode;mode=438}mode|=8192;return FS.mknod(path,mode,dev)},symlink(oldpath,newpath){if(!PATH_FS.resolve(oldpath)){throw new FS.ErrnoError(44)}var lookup=FS.lookupPath(newpath,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var newname=PATH.basename(newpath);var errCode=FS.mayCreate(parent,newname);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.symlink){throw new FS.ErrnoError(63)}return parent.node_ops.symlink(parent,newname,oldpath)},rename(old_path,new_path){var old_dirname=PATH.dirname(old_path);var new_dirname=PATH.dirname(new_path);var old_name=PATH.basename(old_path);var new_name=PATH.basename(new_path);var lookup,old_dir,new_dir;lookup=FS.lookupPath(old_path,{parent:true});old_dir=lookup.node;lookup=FS.lookupPath(new_path,{parent:true});new_dir=lookup.node;if(!old_dir||!new_dir)throw new FS.ErrnoError(44);if(old_dir.mount!==new_dir.mount){throw new FS.ErrnoError(75)}var old_node=FS.lookupNode(old_dir,old_name);var relative=PATH_FS.relative(old_path,new_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(28)}relative=PATH_FS.relative(new_path,old_dirname);if(relative.charAt(0)!=="."){throw new FS.ErrnoError(55)}var new_node;try{new_node=FS.lookupNode(new_dir,new_name)}catch(e){}if(old_node===new_node){return}var isdir=FS.isDir(old_node.mode);var errCode=FS.mayDelete(old_dir,old_name,isdir);if(errCode){throw new FS.ErrnoError(errCode)}errCode=new_node?FS.mayDelete(new_dir,new_name,isdir):FS.mayCreate(new_dir,new_name);if(errCode){throw new FS.ErrnoError(errCode)}if(!old_dir.node_ops.rename){throw new FS.ErrnoError(63)}if(FS.isMountpoint(old_node)||new_node&&FS.isMountpoint(new_node)){throw new FS.ErrnoError(10)}if(new_dir!==old_dir){errCode=FS.nodePermissions(old_dir,"w");if(errCode){throw new FS.ErrnoError(errCode)}}FS.hashRemoveNode(old_node);try{old_dir.node_ops.rename(old_node,new_dir,new_name);old_node.parent=new_dir}catch(e){throw e}finally{FS.hashAddNode(old_node)}},rmdir(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,true);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.rmdir){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.rmdir(parent,name);FS.destroyNode(node)},readdir(path){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var readdir=FS.checkOpExists(node.node_ops.readdir,54);return readdir(node)},unlink(path){var lookup=FS.lookupPath(path,{parent:true});var parent=lookup.node;if(!parent){throw new FS.ErrnoError(44)}var name=PATH.basename(path);var node=FS.lookupNode(parent,name);var errCode=FS.mayDelete(parent,name,false);if(errCode){throw new FS.ErrnoError(errCode)}if(!parent.node_ops.unlink){throw new FS.ErrnoError(63)}if(FS.isMountpoint(node)){throw new FS.ErrnoError(10)}parent.node_ops.unlink(parent,name);FS.destroyNode(node)},readlink(path){var lookup=FS.lookupPath(path);var link=lookup.node;if(!link){throw new FS.ErrnoError(44)}if(!link.node_ops.readlink){throw new FS.ErrnoError(28)}return link.node_ops.readlink(link)},stat(path,dontFollow){var lookup=FS.lookupPath(path,{follow:!dontFollow});var node=lookup.node;var getattr=FS.checkOpExists(node.node_ops.getattr,63);return getattr(node)},fstat(fd){var stream=FS.getStreamChecked(fd);var node=stream.node;var getattr=stream.stream_ops.getattr;var arg=getattr?stream:node;getattr??=node.node_ops.getattr;FS.checkOpExists(getattr,63);return getattr(arg)},lstat(path){return FS.stat(path,true)},doChmod(stream,node,mode,dontFollow){FS.doSetAttr(stream,node,{mode:mode&4095|node.mode&~4095,ctime:Date.now(),dontFollow})},chmod(path,mode,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChmod(null,node,mode,dontFollow)},lchmod(path,mode){FS.chmod(path,mode,true)},fchmod(fd,mode){var stream=FS.getStreamChecked(fd);FS.doChmod(stream,stream.node,mode,false)},doChown(stream,node,dontFollow){FS.doSetAttr(stream,node,{timestamp:Date.now(),dontFollow})},chown(path,uid,gid,dontFollow){var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:!dontFollow});node=lookup.node}else{node=path}FS.doChown(null,node,dontFollow)},lchown(path,uid,gid){FS.chown(path,uid,gid,true)},fchown(fd,uid,gid){var stream=FS.getStreamChecked(fd);FS.doChown(stream,stream.node,false)},doTruncate(stream,node,len){if(FS.isDir(node.mode)){throw new FS.ErrnoError(31)}if(!FS.isFile(node.mode)){throw new FS.ErrnoError(28)}var errCode=FS.nodePermissions(node,"w");if(errCode){throw new FS.ErrnoError(errCode)}FS.doSetAttr(stream,node,{size:len,timestamp:Date.now()})},truncate(path,len){if(len<0){throw new FS.ErrnoError(28)}var node;if(typeof path=="string"){var lookup=FS.lookupPath(path,{follow:true});node=lookup.node}else{node=path}FS.doTruncate(null,node,len)},ftruncate(fd,len){var stream=FS.getStreamChecked(fd);if(len<0||(stream.flags&2097155)===0){throw new FS.ErrnoError(28)}FS.doTruncate(stream,stream.node,len)},utime(path,atime,mtime){var lookup=FS.lookupPath(path,{follow:true});var node=lookup.node;var setattr=FS.checkOpExists(node.node_ops.setattr,63);setattr(node,{atime,mtime})},open(path,flags,mode=438){if(path===""){throw new FS.ErrnoError(44)}flags=typeof flags=="string"?FS_modeStringToFlags(flags):flags;if(flags&64){mode=mode&4095|32768}else{mode=0}var node;var isDirPath;if(typeof path=="object"){node=path}else{isDirPath=path.endsWith("/");var lookup=FS.lookupPath(path,{follow:!(flags&131072),noent_okay:true});node=lookup.node;path=lookup.path}var created=false;if(flags&64){if(node){if(flags&128){throw new FS.ErrnoError(20)}}else if(isDirPath){throw new FS.ErrnoError(31)}else{node=FS.mknod(path,mode|511,0);created=true}}if(!node){throw new FS.ErrnoError(44)}if(FS.isChrdev(node.mode)){flags&=~512}if(flags&65536&&!FS.isDir(node.mode)){throw new FS.ErrnoError(54)}if(!created){var errCode=FS.mayOpen(node,flags);if(errCode){throw new FS.ErrnoError(errCode)}}if(flags&512&&!created){FS.truncate(node,0)}flags&=~(128|512|131072);var stream=FS.createStream({node,path:FS.getPath(node),flags,seekable:true,position:0,stream_ops:node.stream_ops,ungotten:[],error:false});if(stream.stream_ops.open){stream.stream_ops.open(stream)}if(created){FS.chmod(node,mode&511)}if(Module["logReadFiles"]&&!(flags&1)){if(!(path in FS.readFiles)){FS.readFiles[path]=1}}return stream},close(stream){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(stream.getdents)stream.getdents=null;try{if(stream.stream_ops.close){stream.stream_ops.close(stream)}}catch(e){throw e}finally{FS.closeStream(stream.fd)}stream.fd=null},isClosed(stream){return stream.fd===null},llseek(stream,offset,whence){if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if(!stream.seekable||!stream.stream_ops.llseek){throw new FS.ErrnoError(70)}if(whence!=0&&whence!=1&&whence!=2){throw new FS.ErrnoError(28)}stream.position=stream.stream_ops.llseek(stream,offset,whence);stream.ungotten=[];return stream.position},read(stream,buffer,offset,length,position){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.read){throw new FS.ErrnoError(28)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesRead=stream.stream_ops.read(stream,buffer,offset,length,position);if(!seeking)stream.position+=bytesRead;return bytesRead},write(stream,buffer,offset,length,position,canOwn){if(length<0||position<0){throw new FS.ErrnoError(28)}if(FS.isClosed(stream)){throw new FS.ErrnoError(8)}if((stream.flags&2097155)===0){throw new FS.ErrnoError(8)}if(FS.isDir(stream.node.mode)){throw new FS.ErrnoError(31)}if(!stream.stream_ops.write){throw new FS.ErrnoError(28)}if(stream.seekable&&stream.flags&1024){FS.llseek(stream,0,2)}var seeking=typeof position!="undefined";if(!seeking){position=stream.position}else if(!stream.seekable){throw new FS.ErrnoError(70)}var bytesWritten=stream.stream_ops.write(stream,buffer,offset,length,position,canOwn);if(!seeking)stream.position+=bytesWritten;return bytesWritten},mmap(stream,length,position,prot,flags){if((prot&2)!==0&&(flags&2)===0&&(stream.flags&2097155)!==2){throw new FS.ErrnoError(2)}if((stream.flags&2097155)===1){throw new FS.ErrnoError(2)}if(!stream.stream_ops.mmap){throw new FS.ErrnoError(43)}if(!length){throw new FS.ErrnoError(28)}return stream.stream_ops.mmap(stream,length,position,prot,flags)},msync(stream,buffer,offset,length,mmapFlags){if(!stream.stream_ops.msync){return 0}return stream.stream_ops.msync(stream,buffer,offset,length,mmapFlags)},ioctl(stream,cmd,arg){if(!stream.stream_ops.ioctl){throw new FS.ErrnoError(59)}return stream.stream_ops.ioctl(stream,cmd,arg)},readFile(path,opts={}){opts.flags=opts.flags||0;opts.encoding=opts.encoding||"binary";if(opts.encoding!=="utf8"&&opts.encoding!=="binary"){abort(`Invalid encoding type "${opts.encoding}"`)}var stream=FS.open(path,opts.flags);var stat=FS.stat(path);var length=stat.size;var buf=new Uint8Array(length);FS.read(stream,buf,0,length,0);if(opts.encoding==="utf8"){buf=UTF8ArrayToString(buf)}FS.close(stream);return buf},writeFile(path,data,opts={}){opts.flags=opts.flags||577;var stream=FS.open(path,opts.flags,opts.mode);if(typeof data=="string"){data=new Uint8Array(intArrayFromString(data,true))}if(ArrayBuffer.isView(data)){FS.write(stream,data,0,data.byteLength,undefined,opts.canOwn)}else{abort("Unsupported data type")}FS.close(stream)},cwd:()=>FS.currentPath,chdir(path){var lookup=FS.lookupPath(path,{follow:true});if(lookup.node===null){throw new FS.ErrnoError(44)}if(!FS.isDir(lookup.node.mode)){throw new FS.ErrnoError(54)}var errCode=FS.nodePermissions(lookup.node,"x");if(errCode){throw new FS.ErrnoError(errCode)}FS.currentPath=lookup.path},createDefaultDirectories(){FS.mkdir("/tmp");FS.mkdir("/home");FS.mkdir("/home/web_user")},createDefaultDevices(){FS.mkdir("/dev");FS.registerDevice(FS.makedev(1,3),{read:()=>0,write:(stream,buffer,offset,length,pos)=>length,llseek:()=>0});FS.mkdev("/dev/null",FS.makedev(1,3));TTY.register(FS.makedev(5,0),TTY.default_tty_ops);TTY.register(FS.makedev(6,0),TTY.default_tty1_ops);FS.mkdev("/dev/tty",FS.makedev(5,0));FS.mkdev("/dev/tty1",FS.makedev(6,0));var randomBuffer=new Uint8Array(1024),randomLeft=0;var randomByte=()=>{if(randomLeft===0){randomFill(randomBuffer);randomLeft=randomBuffer.byteLength}return randomBuffer[--randomLeft]};FS.createDevice("/dev","random",randomByte);FS.createDevice("/dev","urandom",randomByte);FS.mkdir("/dev/shm");FS.mkdir("/dev/shm/tmp")},createSpecialDirectories(){FS.mkdir("/proc");var proc_self=FS.mkdir("/proc/self");FS.mkdir("/proc/self/fd");FS.mount({mount(){var node=FS.createNode(proc_self,"fd",16895,73);node.stream_ops={llseek:MEMFS.stream_ops.llseek};node.node_ops={lookup(parent,name){var fd=+name;var stream=FS.getStreamChecked(fd);var ret={parent:null,mount:{mountpoint:"fake"},node_ops:{readlink:()=>stream.path},id:fd+1};ret.parent=ret;return ret},readdir(){return Array.from(FS.streams.entries()).filter(([k,v])=>v).map(([k,v])=>k.toString())}};return node}},{},"/proc/self/fd")},createStandardStreams(input,output,error){if(input){FS.createDevice("/dev","stdin",input)}else{FS.symlink("/dev/tty","/dev/stdin")}if(output){FS.createDevice("/dev","stdout",null,output)}else{FS.symlink("/dev/tty","/dev/stdout")}if(error){FS.createDevice("/dev","stderr",null,error)}else{FS.symlink("/dev/tty1","/dev/stderr")}var stdin=FS.open("/dev/stdin",0);var stdout=FS.open("/dev/stdout",1);var stderr=FS.open("/dev/stderr",1)},staticInit(){FS.nameTable=new Array(4096);FS.mount(MEMFS,{},"/");FS.createDefaultDirectories();FS.createDefaultDevices();FS.createSpecialDirectories();FS.filesystems={MEMFS}},init(input,output,error){FS.initialized=true;input??=Module["stdin"];output??=Module["stdout"];error??=Module["stderr"];FS.createStandardStreams(input,output,error)},quit(){FS.initialized=false;for(var stream of FS.streams){if(stream){FS.close(stream)}}},findObject(path,dontResolveLastLink){var ret=FS.analyzePath(path,dontResolveLastLink);if(!ret.exists){return null}return ret.object},analyzePath(path,dontResolveLastLink){try{var lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});path=lookup.path}catch(e){}var ret={isRoot:false,exists:false,error:0,name:null,path:null,object:null,parentExists:false,parentPath:null,parentObject:null};try{var lookup=FS.lookupPath(path,{parent:true});ret.parentExists=true;ret.parentPath=lookup.path;ret.parentObject=lookup.node;ret.name=PATH.basename(path);lookup=FS.lookupPath(path,{follow:!dontResolveLastLink});ret.exists=true;ret.path=lookup.path;ret.object=lookup.node;ret.name=lookup.node.name;ret.isRoot=lookup.path==="/"}catch(e){ret.error=e.errno}return ret},createPath(parent,path,canRead,canWrite){parent=typeof parent=="string"?parent:FS.getPath(parent);var parts=path.split("/").reverse();while(parts.length){var part=parts.pop();if(!part)continue;var current=PATH.join2(parent,part);try{FS.mkdir(current)}catch(e){if(e.errno!=20)throw e}parent=current}return current},createFile(parent,name,properties,canRead,canWrite){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(canRead,canWrite);return FS.create(path,mode)},createDataFile(parent,name,data,canRead,canWrite,canOwn){var path=name;if(parent){parent=typeof parent=="string"?parent:FS.getPath(parent);path=name?PATH.join2(parent,name):parent}var mode=FS_getMode(canRead,canWrite);var node=FS.create(path,mode);if(data){if(typeof data=="string"){var arr=new Array(data.length);for(var i=0,len=data.length;i<len;++i)arr[i]=data.charCodeAt(i);data=arr}FS.chmod(node,mode|146);var stream=FS.open(node,577);FS.write(stream,data,0,data.length,0,canOwn);FS.close(stream);FS.chmod(node,mode)}},createDevice(parent,name,input,output){var path=PATH.join2(typeof parent=="string"?parent:FS.getPath(parent),name);var mode=FS_getMode(!!input,!!output);FS.createDevice.major??=64;var dev=FS.makedev(FS.createDevice.major++,0);FS.registerDevice(dev,{open(stream){stream.seekable=false},close(stream){if(output?.buffer?.length){output(10)}},read(stream,buffer,offset,length,pos){var bytesRead=0;for(var i=0;i<length;i++){var result;try{result=input()}catch(e){throw new FS.ErrnoError(29)}if(result===undefined&&bytesRead===0){throw new FS.ErrnoError(6)}if(result===null||result===undefined)break;bytesRead++;buffer[offset+i]=result}if(bytesRead){stream.node.atime=Date.now()}return bytesRead},write(stream,buffer,offset,length,pos){for(var i=0;i<length;i++){try{output(buffer[offset+i])}catch(e){throw new FS.ErrnoError(29)}}if(length){stream.node.mtime=stream.node.ctime=Date.now()}return i}});return FS.mkdev(path,mode,dev)},forceLoadFile(obj){if(obj.isDevice||obj.isFolder||obj.link||obj.contents)return true;if(globalThis.XMLHttpRequest){abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.")}else{try{obj.contents=readBinary(obj.url)}catch(e){throw new FS.ErrnoError(29)}}},createLazyFile(parent,name,url,canRead,canWrite){class LazyUint8Array{lengthKnown=false;chunks=[];get(idx){if(idx>this.length-1||idx<0){return undefined}var chunkOffset=idx%this.chunkSize;var chunkNum=idx/this.chunkSize|0;return this.getter(chunkNum)[chunkOffset]}setDataGetter(getter){this.getter=getter}cacheLength(){var xhr=new XMLHttpRequest;xhr.open("HEAD",url,false);xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);var datalength=Number(xhr.getResponseHeader("Content-length"));var header;var hasByteServing=(header=xhr.getResponseHeader("Accept-Ranges"))&&header==="bytes";var usesGzip=(header=xhr.getResponseHeader("Content-Encoding"))&&header==="gzip";var chunkSize=1024*1024;if(!hasByteServing)chunkSize=datalength;var doXHR=(from,to)=>{if(from>to)abort("invalid range ("+from+", "+to+") or no bytes requested!");if(to>datalength-1)abort("only "+datalength+" bytes available! programmer error!");var xhr=new XMLHttpRequest;xhr.open("GET",url,false);if(datalength!==chunkSize)xhr.setRequestHeader("Range","bytes="+from+"-"+to);xhr.responseType="arraybuffer";if(xhr.overrideMimeType){xhr.overrideMimeType("text/plain; charset=x-user-defined")}xhr.send(null);if(!(xhr.status>=200&&xhr.status<300||xhr.status===304))abort("Couldn\'t load "+url+". Status: "+xhr.status);if(xhr.response!==undefined){return new Uint8Array(xhr.response||[])}return intArrayFromString(xhr.responseText||"",true)};var lazyArray=this;lazyArray.setDataGetter(chunkNum=>{var start=chunkNum*chunkSize;var end=(chunkNum+1)*chunkSize-1;end=Math.min(end,datalength-1);if(typeof lazyArray.chunks[chunkNum]=="undefined"){lazyArray.chunks[chunkNum]=doXHR(start,end)}if(typeof lazyArray.chunks[chunkNum]=="undefined")abort("doXHR failed!");return lazyArray.chunks[chunkNum]});if(usesGzip||!datalength){chunkSize=datalength=1;datalength=this.getter(0).length;chunkSize=datalength;out("LazyFiles on gzip forces download of the whole file when length is accessed")}this._length=datalength;this._chunkSize=chunkSize;this.lengthKnown=true}get length(){if(!this.lengthKnown){this.cacheLength()}return this._length}get chunkSize(){if(!this.lengthKnown){this.cacheLength()}return this._chunkSize}}if(globalThis.XMLHttpRequest){if(!ENVIRONMENT_IS_WORKER)abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");var lazyArray=new LazyUint8Array;var properties={isDevice:false,contents:lazyArray}}else{var properties={isDevice:false,url}}var node=FS.createFile(parent,name,properties,canRead,canWrite);if(properties.contents){node.contents=properties.contents}else if(properties.url){node.contents=null;node.url=properties.url}Object.defineProperties(node,{usedBytes:{get:function(){return this.contents.length}}});var stream_ops={};for(const[key,fn]of Object.entries(node.stream_ops)){stream_ops[key]=(...args)=>{FS.forceLoadFile(node);return fn(...args)}}function writeChunks(stream,buffer,offset,length,position){var contents=stream.node.contents;if(position>=contents.length)return 0;var size=Math.min(contents.length-position,length);if(contents.slice){for(var i=0;i<size;i++){buffer[offset+i]=contents[position+i]}}else{for(var i=0;i<size;i++){buffer[offset+i]=contents.get(position+i)}}return size}stream_ops.read=(stream,buffer,offset,length,position)=>{FS.forceLoadFile(node);return writeChunks(stream,buffer,offset,length,position)};stream_ops.mmap=(stream,length,position,prot,flags)=>{FS.forceLoadFile(node);var ptr=mmapAlloc(length);if(!ptr){throw new FS.ErrnoError(48)}writeChunks(stream,(growMemViews(),HEAP8),ptr,length,position);return{ptr,allocated:true}};node.stream_ops=stream_ops;return node}};var UTF8ToString=(ptr,maxBytesToRead,ignoreNul)=>ptr?UTF8ArrayToString((growMemViews(),HEAPU8),ptr,maxBytesToRead,ignoreNul):"";var SYSCALLS={DEFAULT_POLLMASK:5,calculateAt(dirfd,path,allowEmpty){if(PATH.isAbs(path)){return path}var dir;if(dirfd===-100){dir=FS.cwd()}else{var dirstream=SYSCALLS.getStreamFromFD(dirfd);dir=dirstream.path}if(path.length==0){if(!allowEmpty){throw new FS.ErrnoError(44)}return dir}return dir+"/"+path},writeStat(buf,stat){(growMemViews(),HEAPU32)[buf/4]=stat.dev;(growMemViews(),HEAPU32)[(buf+4)/4]=stat.mode;(growMemViews(),HEAPU64)[(buf+8)/8]=BigInt(stat.nlink);(growMemViews(),HEAPU32)[(buf+16)/4]=stat.uid;(growMemViews(),HEAPU32)[(buf+20)/4]=stat.gid;(growMemViews(),HEAPU32)[(buf+24)/4]=stat.rdev;(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stat.size);(growMemViews(),HEAP32)[(buf+40)/4]=4096;(growMemViews(),HEAP32)[(buf+44)/4]=stat.blocks;var atime=stat.atime.getTime();var mtime=stat.mtime.getTime();var ctime=stat.ctime.getTime();(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(Math.floor(atime/1e3));(growMemViews(),HEAPU64)[(buf+56)/8]=BigInt(atime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+64)/8]=BigInt(Math.floor(mtime/1e3));(growMemViews(),HEAPU64)[(buf+72)/8]=BigInt(mtime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+80)/8]=BigInt(Math.floor(ctime/1e3));(growMemViews(),HEAPU64)[(buf+88)/8]=BigInt(ctime%1e3*1e3*1e3);(growMemViews(),HEAP64)[(buf+96)/8]=BigInt(stat.ino);return 0},writeStatFs(buf,stats){(growMemViews(),HEAPU32)[(buf+8)/4]=stats.bsize;(growMemViews(),HEAPU32)[(buf+72)/4]=stats.bsize;(growMemViews(),HEAP64)[(buf+16)/8]=BigInt(stats.blocks);(growMemViews(),HEAP64)[(buf+24)/8]=BigInt(stats.bfree);(growMemViews(),HEAP64)[(buf+32)/8]=BigInt(stats.bavail);(growMemViews(),HEAP64)[(buf+40)/8]=BigInt(stats.files);(growMemViews(),HEAP64)[(buf+48)/8]=BigInt(stats.ffree);(growMemViews(),HEAPU32)[(buf+56)/4]=stats.fsid;(growMemViews(),HEAPU32)[(buf+80)/4]=stats.flags;(growMemViews(),HEAPU32)[(buf+64)/4]=stats.namelen},doMsync(addr,stream,len,flags,offset){if(!FS.isFile(stream.node.mode)){throw new FS.ErrnoError(43)}if(flags&2){return 0}var buffer=(growMemViews(),HEAPU8).slice(addr,addr+len);FS.msync(stream,buffer,offset,len,flags)},getStreamFromFD(fd){var stream=FS.getStreamChecked(fd);return stream},varargs:undefined,getStr(ptr){var ret=UTF8ToString(ptr);return ret}};function ___syscall_fcntl64(fd,cmd,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(3,0,1,fd,cmd,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(cmd){case 0:{var arg=syscallGetVarargI();if(arg<0){return-28}while(FS.streams[arg]){arg++}var newStream;newStream=FS.dupStream(stream,arg);return newStream.fd}case 1:case 2:return 0;case 3:return stream.flags;case 4:{var arg=syscallGetVarargI();stream.flags|=arg;return 0}case 5:{var arg=syscallGetVarargP();var offset=0;(growMemViews(),HEAP16)[(arg+offset)/2]=2;return 0}case 6:case 7:return 0}return-28}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_fstat64(fd,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(4,0,1,fd,buf);buf=bigintToI53Checked(buf);try{return SYSCALLS.writeStat(buf,FS.fstat(fd))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var stringToUTF8=(str,outPtr,maxBytesToWrite)=>stringToUTF8Array(str,(growMemViews(),HEAPU8),outPtr,maxBytesToWrite);function ___syscall_getcwd(buf,size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(5,0,1,buf,size);buf=bigintToI53Checked(buf);size=bigintToI53Checked(size);try{if(size===0)return-28;var cwd=FS.cwd();var cwdLengthInBytes=lengthBytesUTF8(cwd)+1;if(size<cwdLengthInBytes)return-68;stringToUTF8(cwd,buf,size);return cwdLengthInBytes}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_getdents64(fd,dirp,count){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(6,0,1,fd,dirp,count);dirp=bigintToI53Checked(dirp);count=bigintToI53Checked(count);try{var stream=SYSCALLS.getStreamFromFD(fd);stream.getdents||=FS.readdir(stream.path);var struct_size=280;var pos=0;var off=FS.llseek(stream,0,1);var startIdx=Math.floor(off/struct_size);var endIdx=Math.min(stream.getdents.length,startIdx+Math.floor(count/struct_size));for(var idx=startIdx;idx<endIdx;idx++){var id;var type;var name=stream.getdents[idx];if(name==="."){id=stream.node.id;type=4}else if(name===".."){var lookup=FS.lookupPath(stream.path,{parent:true});id=lookup.node.id;type=4}else{var child;try{child=FS.lookupNode(stream.node,name)}catch(e){if(e?.errno===28){continue}throw e}id=child.id;type=FS.isChrdev(child.mode)?2:FS.isDir(child.mode)?4:FS.isLink(child.mode)?10:8}(growMemViews(),HEAP64)[(dirp+pos)/8]=BigInt(id);(growMemViews(),HEAP64)[(dirp+pos+8)/8]=BigInt((idx+1)*struct_size);(growMemViews(),HEAP16)[(dirp+pos+16)/2]=280;(growMemViews(),HEAP8)[dirp+pos+18]=type;stringToUTF8(name,dirp+pos+19,256);pos+=struct_size}FS.llseek(stream,idx*struct_size,0);return pos}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_ioctl(fd,op,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(7,0,1,fd,op,varargs);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{var stream=SYSCALLS.getStreamFromFD(fd);switch(op){case 21509:{if(!stream.tty)return-59;return 0}case 21505:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcgets){var termios=stream.tty.ops.ioctl_tcgets(stream);var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=termios.c_iflag||0;(growMemViews(),HEAP32)[(argp+4)/4]=termios.c_oflag||0;(growMemViews(),HEAP32)[(argp+8)/4]=termios.c_cflag||0;(growMemViews(),HEAP32)[(argp+12)/4]=termios.c_lflag||0;for(var i=0;i<32;i++){(growMemViews(),HEAP8)[argp+i+17]=termios.c_cc[i]||0}return 0}return 0}case 21510:case 21511:case 21512:{if(!stream.tty)return-59;return 0}case 21506:case 21507:case 21508:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tcsets){var argp=syscallGetVarargP();var c_iflag=(growMemViews(),HEAP32)[argp/4];var c_oflag=(growMemViews(),HEAP32)[(argp+4)/4];var c_cflag=(growMemViews(),HEAP32)[(argp+8)/4];var c_lflag=(growMemViews(),HEAP32)[(argp+12)/4];var c_cc=[];for(var i=0;i<32;i++){c_cc.push((growMemViews(),HEAP8)[argp+i+17])}return stream.tty.ops.ioctl_tcsets(stream.tty,op,{c_iflag,c_oflag,c_cflag,c_lflag,c_cc})}return 0}case 21519:{if(!stream.tty)return-59;var argp=syscallGetVarargP();(growMemViews(),HEAP32)[argp/4]=0;return 0}case 21520:{if(!stream.tty)return-59;return-28}case 21537:case 21531:{var argp=syscallGetVarargP();return FS.ioctl(stream,op,argp)}case 21523:{if(!stream.tty)return-59;if(stream.tty.ops.ioctl_tiocgwinsz){var winsize=stream.tty.ops.ioctl_tiocgwinsz(stream.tty);var argp=syscallGetVarargP();(growMemViews(),HEAP16)[argp/2]=winsize[0];(growMemViews(),HEAP16)[(argp+2)/2]=winsize[1]}return 0}case 21524:{if(!stream.tty)return-59;return 0}case 21515:{if(!stream.tty)return-59;return 0}default:return-28}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_lstat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(8,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.lstat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_newfstatat(dirfd,path,buf,flags){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(9,0,1,dirfd,path,buf,flags);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);var nofollow=flags&256;var allowEmpty=flags&4096;flags=flags&~6400;path=SYSCALLS.calculateAt(dirfd,path,allowEmpty);return SYSCALLS.writeStat(buf,nofollow?FS.lstat(path):FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_openat(dirfd,path,flags,varargs){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(10,0,1,dirfd,path,flags,varargs);path=bigintToI53Checked(path);varargs=bigintToI53Checked(varargs);SYSCALLS.varargs=varargs;try{path=SYSCALLS.getStr(path);path=SYSCALLS.calculateAt(dirfd,path);var mode=varargs?syscallGetVarargI():0;return FS.open(path,flags,mode).fd}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function ___syscall_stat64(path,buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(11,0,1,path,buf);path=bigintToI53Checked(path);buf=bigintToI53Checked(buf);try{path=SYSCALLS.getStr(path);return SYSCALLS.writeStat(buf,FS.stat(path))}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __abort_js=()=>abort("");function __emscripten_init_main_thread_js(tb){tb=bigintToI53Checked(tb);__emscripten_thread_init(tb,!ENVIRONMENT_IS_WORKER,1,!ENVIRONMENT_IS_WEB,5242880,false);PThread.threadInitTLS()}var handleException=e=>{if(e instanceof ExitStatus||e=="unwind"){return EXITSTATUS}quit_(1,e)};var maybeExit=()=>{if(!keepRuntimeAlive()){try{if(ENVIRONMENT_IS_PTHREAD){if(_pthread_self())__emscripten_thread_exit(EXITSTATUS);return}_exit(EXITSTATUS)}catch(e){handleException(e)}}};var callUserCallback=func=>{if(ABORT){return}try{func();maybeExit()}catch(e){handleException(e)}};function __emscripten_thread_mailbox_await(pthread_ptr){pthread_ptr=bigintToI53Checked(pthread_ptr);if(Atomics.waitAsync){var wait=Atomics.waitAsync((growMemViews(),HEAP32),pthread_ptr/4,pthread_ptr);wait.value.then(checkMailbox);var waitingAsync=pthread_ptr+228;Atomics.store((growMemViews(),HEAP32),waitingAsync/4,1)}}var checkMailbox=()=>callUserCallback(()=>{var pthread_ptr=_pthread_self();if(pthread_ptr){__emscripten_thread_mailbox_await(pthread_ptr);__emscripten_check_mailbox()}});function __emscripten_notify_mailbox_postmessage(targetThread,currThreadId){targetThread=bigintToI53Checked(targetThread);currThreadId=bigintToI53Checked(currThreadId);if(targetThread==currThreadId){setTimeout(checkMailbox)}else if(ENVIRONMENT_IS_PTHREAD){postMessage({targetThread,cmd:"checkMailbox"})}else{var worker=PThread.pthreads[targetThread];if(!worker){return}worker.postMessage({cmd:"checkMailbox"})}}var proxiedJSCallArgs=[];function __emscripten_receive_on_main_thread_js(funcIndex,emAsmAddr,callingThread,numCallArgs,args){emAsmAddr=bigintToI53Checked(emAsmAddr);callingThread=bigintToI53Checked(callingThread);args=bigintToI53Checked(args);numCallArgs/=2;proxiedJSCallArgs.length=numCallArgs;var b=args/8;for(var i=0;i<numCallArgs;i++){if((growMemViews(),HEAP64)[b+2*i]){proxiedJSCallArgs[i]=(growMemViews(),HEAP64)[b+2*i+1]}else{proxiedJSCallArgs[i]=(growMemViews(),HEAPF64)[b+2*i+1]}}var func=proxiedFunctionTable[funcIndex];PThread.currentProxiedOperationCallerThread=callingThread;var rtn=func(...proxiedJSCallArgs);PThread.currentProxiedOperationCallerThread=0;if(typeof rtn=="bigint"){rtn=bigintToI53Checked(rtn)}return rtn}function __emscripten_thread_cleanup(thread){thread=bigintToI53Checked(thread);if(!ENVIRONMENT_IS_PTHREAD)cleanupThread(thread);else postMessage({cmd:"cleanupThread",thread})}function __emscripten_thread_set_strongref(thread){thread=bigintToI53Checked(thread);if(ENVIRONMENT_IS_NODE){PThread.pthreads[thread].ref()}}var isLeapYear=year=>year%4===0&&(year%100!==0||year%400===0);var MONTH_DAYS_LEAP_CUMULATIVE=[0,31,60,91,121,152,182,213,244,274,305,335];var MONTH_DAYS_REGULAR_CUMULATIVE=[0,31,59,90,120,151,181,212,243,273,304,334];var ydayFromDate=date=>{var leap=isLeapYear(date.getFullYear());var monthDaysCumulative=leap?MONTH_DAYS_LEAP_CUMULATIVE:MONTH_DAYS_REGULAR_CUMULATIVE;var yday=monthDaysCumulative[date.getMonth()]+date.getDate()-1;return yday};function __localtime_js(time,tmPtr){time=bigintToI53Checked(time);tmPtr=bigintToI53Checked(tmPtr);var date=new Date(time*1e3);(growMemViews(),HEAP32)[tmPtr/4]=date.getSeconds();(growMemViews(),HEAP32)[(tmPtr+4)/4]=date.getMinutes();(growMemViews(),HEAP32)[(tmPtr+8)/4]=date.getHours();(growMemViews(),HEAP32)[(tmPtr+12)/4]=date.getDate();(growMemViews(),HEAP32)[(tmPtr+16)/4]=date.getMonth();(growMemViews(),HEAP32)[(tmPtr+20)/4]=date.getFullYear()-1900;(growMemViews(),HEAP32)[(tmPtr+24)/4]=date.getDay();var yday=ydayFromDate(date)|0;(growMemViews(),HEAP32)[(tmPtr+28)/4]=yday;(growMemViews(),HEAP64)[(tmPtr+40)/8]=BigInt(-(date.getTimezoneOffset()*60));var start=new Date(date.getFullYear(),0,1);var summerOffset=new Date(date.getFullYear(),6,1).getTimezoneOffset();var winterOffset=start.getTimezoneOffset();var dst=(summerOffset!=winterOffset&&date.getTimezoneOffset()==Math.min(winterOffset,summerOffset))|0;(growMemViews(),HEAP32)[(tmPtr+32)/4]=dst}function __mmap_js(len,prot,flags,fd,offset,allocated,addr){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(12,0,1,len,prot,flags,fd,offset,allocated,addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);allocated=bigintToI53Checked(allocated);addr=bigintToI53Checked(addr);try{var stream=SYSCALLS.getStreamFromFD(fd);var res=FS.mmap(stream,len,offset,prot,flags);var ptr=res.ptr;(growMemViews(),HEAP32)[allocated/4]=res.allocated;(growMemViews(),HEAPU64)[addr/8]=BigInt(ptr);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}function __munmap_js(addr,len,prot,flags,fd,offset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(13,0,1,addr,len,prot,flags,fd,offset);addr=bigintToI53Checked(addr);len=bigintToI53Checked(len);offset=bigintToI53Checked(offset);try{var stream=SYSCALLS.getStreamFromFD(fd);if(prot&2){SYSCALLS.doMsync(addr,stream,len,flags,offset)}}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return-e.errno}}var __tzset_js=function(timezone,daylight,std_name,dst_name){timezone=bigintToI53Checked(timezone);daylight=bigintToI53Checked(daylight);std_name=bigintToI53Checked(std_name);dst_name=bigintToI53Checked(dst_name);var currentYear=(new Date).getFullYear();var winter=new Date(currentYear,0,1);var summer=new Date(currentYear,6,1);var winterOffset=winter.getTimezoneOffset();var summerOffset=summer.getTimezoneOffset();var stdTimezoneOffset=Math.max(winterOffset,summerOffset);(growMemViews(),HEAPU64)[timezone/8]=BigInt(stdTimezoneOffset*60);(growMemViews(),HEAP32)[daylight/4]=Number(winterOffset!=summerOffset);var extractZone=timezoneOffset=>{var sign=timezoneOffset>=0?"-":"+";var absOffset=Math.abs(timezoneOffset);var hours=String(Math.floor(absOffset/60)).padStart(2,"0");var minutes=String(absOffset%60).padStart(2,"0");return`UTC${sign}${hours}${minutes}`};var winterName=extractZone(winterOffset);var summerName=extractZone(summerOffset);if(summerOffset<winterOffset){stringToUTF8(winterName,std_name,17);stringToUTF8(summerName,dst_name,17)}else{stringToUTF8(winterName,dst_name,17);stringToUTF8(summerName,std_name,17)}};var _emscripten_get_now=()=>performance.timeOrigin+performance.now();var _emscripten_date_now=()=>Date.now();var nowIsMonotonic=1;var checkWasiClock=clock_id=>clock_id>=0&&clock_id<=3;function _clock_time_get(clk_id,ignored_precision,ptime){ignored_precision=bigintToI53Checked(ignored_precision);ptime=bigintToI53Checked(ptime);if(!checkWasiClock(clk_id)){return 28}var now;if(clk_id===0){now=_emscripten_date_now()}else if(nowIsMonotonic){now=_emscripten_get_now()}else{return 52}var nsec=Math.round(now*1e3*1e3);(growMemViews(),HEAP64)[ptime/8]=BigInt(nsec);return 0}var _emscripten_check_blocking_allowed=()=>{};var runtimeKeepalivePush=()=>{runtimeKeepaliveCounter+=1};var _emscripten_exit_with_live_runtime=()=>{runtimeKeepalivePush();throw"unwind"};var jsStackTrace=()=>(new Error).stack.toString();var getCallstack=flags=>{var callstack=jsStackTrace();var lines=callstack.split("\\n");callstack="";var firefoxRe=new RegExp("\\\\s*(.*?)@(.*?):([0-9]+):([0-9]+)");var chromeRe=new RegExp("\\\\s*at (.*?) \\\\((.*):(.*):(.*)\\\\)");for(var line of lines){var symbolName="";var file="";var lineno=0;var column=0;var parts=chromeRe.exec(line);if(parts?.length==5){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]}else{parts=firefoxRe.exec(line);if(parts?.length>=4){symbolName=parts[1];file=parts[2];lineno=parts[3];column=parts[4]|0}else{callstack+=line+"\\n";continue}}if(symbolName=="_emscripten_log"||symbolName=="_emscripten_get_callstack"){callstack="";continue}if(flags&24){if(flags&64){file=file.substring(file.replace(/\\\\/g,"/").lastIndexOf("/")+1)}callstack+=`    at ${symbolName} (${file}:${lineno}:${column})\\n`}}callstack=callstack.replace(/\\s+$/,"");return callstack};function _emscripten_get_callstack(flags,str,maxbytes){str=bigintToI53Checked(str);var callstack=getCallstack(flags);if(!str||maxbytes<=0){return lengthBytesUTF8(callstack)+1}var bytesWrittenExcludingNull=stringToUTF8(callstack,str,maxbytes);return bytesWrittenExcludingNull+1}var getHeapMax=()=>4294967296;var _emscripten_get_heap_max=()=>BigInt(getHeapMax());var _emscripten_has_asyncify=()=>2;var _emscripten_num_logical_cores=()=>ENVIRONMENT_IS_NODE?require("os").cpus().length:navigator["hardwareConcurrency"];var growMemory=size=>{var oldHeapSize=wasmMemory.buffer.byteLength;var pages=(size-oldHeapSize+65535)/65536|0;try{wasmMemory.grow(BigInt(pages));updateMemoryViews();return 1}catch(e){}};function _emscripten_resize_heap(requestedSize){requestedSize=bigintToI53Checked(requestedSize);var oldSize=(growMemViews(),HEAPU8).length;if(requestedSize<=oldSize){return false}var maxHeapSize=getHeapMax();if(requestedSize>maxHeapSize){return false}for(var cutDown=1;cutDown<=4;cutDown*=2){var overGrownHeapSize=oldSize*(1+.2/cutDown);overGrownHeapSize=Math.min(overGrownHeapSize,requestedSize+100663296);var newSize=Math.min(maxHeapSize,alignMemory(Math.max(requestedSize,overGrownHeapSize),65536));var replacement=growMemory(newSize);if(replacement){return true}}return false}var stringToUTF8OnStack=str=>{var size=lengthBytesUTF8(str)+1;var ret=stackAlloc(size);stringToUTF8(str,ret,size);return ret};var writeI53ToI64=(ptr,num)=>{(growMemViews(),HEAPU32)[ptr/4]=num;var lower=(growMemViews(),HEAPU32)[ptr/4];(growMemViews(),HEAPU32)[(ptr+4)/4]=(num-lower)/4294967296};var stringToNewUTF8=str=>{var size=lengthBytesUTF8(str)+1;var ret=_malloc(size);if(ret)stringToUTF8(str,ret,size);return ret};var readI53FromI64=ptr=>(growMemViews(),HEAPU32)[ptr/4]+(growMemViews(),HEAP32)[(ptr+4)/4]*4294967296;var WebGPU={Internals:{jsObjects:[],jsObjectInsert:(ptr,jsObject)=>{WebGPU.Internals.jsObjects[ptr]=jsObject},bufferOnUnmaps:[],futures:[],futureInsert:(futureId,promise)=>{WebGPU.Internals.futures[futureId]=new Promise(resolve=>promise.finally(()=>resolve(futureId)))}},getJsObject:ptr=>{if(!ptr)return undefined;return WebGPU.Internals.jsObjects[ptr]},importJsAdapter:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateAdapter(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroup:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroup(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBindGroupLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateBindGroupLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsBuffer:(buffer,parentPtr=0)=>{assert(buffer.mapState==="unmapped");var bufferPtr=_emwgpuCreateBuffer(parentPtr);WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);return bufferPtr},importJsCommandBuffer:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandBuffer(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsCommandEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateCommandEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsComputePipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateComputePipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsDevice:(device,parentPtr=0)=>{var queuePtr=_emwgpuCreateQueue(parentPtr);var devicePtr=_emwgpuCreateDevice(parentPtr,queuePtr);WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);return devicePtr},importJsExternalTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateExternalTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsPipelineLayout:(obj,parentPtr=0)=>{var ptr=_emwgpuCreatePipelineLayout(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQuerySet:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQuerySet(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsQueue:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateQueue(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundle:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundle(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderBundleEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderBundleEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPassEncoder:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPassEncoder(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsRenderPipeline:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateRenderPipeline(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSampler:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSampler(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsShaderModule:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateShaderModule(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsSurface:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateSurface(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTexture:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTexture(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},importJsTextureView:(obj,parentPtr=0)=>{var ptr=_emwgpuCreateTextureView(parentPtr);WebGPU.Internals.jsObjects[ptr]=obj;return ptr},errorCallback:(callback,type,message,userdata)=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(message);((a1,a2,a3)=>getWasmTableEntry(callback).call(null,a1,BigInt(a2),BigInt(a3)))(type,BigInt(messagePtr),userdata);stackRestore(sp)},iterateExtensions:(root,handlers)=>{for(var ptr=Number((growMemViews(),HEAPU64)[root/8]);ptr;ptr=Number((growMemViews(),HEAPU64)[ptr/8])){var sType=(growMemViews(),HEAP32)[(ptr+8)/4];var handler=handlers[sType](ptr)}},setStringView:(ptr,data,length)=>{(growMemViews(),HEAPU64)[ptr/8]=BigInt(data);(growMemViews(),HEAPU64)[(ptr+8)/8]=BigInt(length)},makeStringFromStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);return UTF8ToString(ptr,length)},makeStringFromOptionalStringView:stringViewPtr=>{var ptr=Number((growMemViews(),HEAPU64)[stringViewPtr/8]);var length=Number((growMemViews(),HEAPU64)[(stringViewPtr+8)/8]);if(!ptr){if(length===0){return""}return undefined}return UTF8ToString(ptr,length)},makeColor:ptr=>({r:(growMemViews(),HEAPF64)[ptr/8],g:(growMemViews(),HEAPF64)[(ptr+8)/8],b:(growMemViews(),HEAPF64)[(ptr+16)/8],a:(growMemViews(),HEAPF64)[(ptr+24)/8]}),makeExtent3D:ptr=>({width:(growMemViews(),HEAPU32)[ptr/4],height:(growMemViews(),HEAPU32)[(ptr+4)/4],depthOrArrayLayers:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeOrigin3D:ptr=>({x:(growMemViews(),HEAPU32)[ptr/4],y:(growMemViews(),HEAPU32)[(ptr+4)/4],z:(growMemViews(),HEAPU32)[(ptr+8)/4]}),makeTexelCopyTextureInfo:ptr=>({texture:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[ptr/8])),mipLevel:(growMemViews(),HEAPU32)[(ptr+8)/4],origin:WebGPU.makeOrigin3D(ptr+12),aspect:WebGPU.TextureAspect[(growMemViews(),HEAP32)[(ptr+24)/4]]}),makeTexelCopyBufferLayout:ptr=>{var bytesPerRow=(growMemViews(),HEAPU32)[(ptr+8)/4];var rowsPerImage=(growMemViews(),HEAPU32)[(ptr+12)/4];return{offset:readI53FromI64(ptr),bytesPerRow:bytesPerRow===4294967295?undefined:bytesPerRow,rowsPerImage:rowsPerImage===4294967295?undefined:rowsPerImage}},makeTexelCopyBufferInfo:ptr=>{var layoutPtr=ptr+0;var bufferCopyView=WebGPU.makeTexelCopyBufferLayout(layoutPtr);bufferCopyView["buffer"]=WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+16)/8]));return bufferCopyView},makePassTimestampWrites:ptr=>{if(ptr===0)return undefined;return{querySet:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),beginningOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+16)/4],endOfPassWriteIndex:(growMemViews(),HEAPU32)[(ptr+20)/4]}},makePipelineConstants:(constantCount,constantsPtr)=>{if(!constantCount)return;var constants={};for(var i=0;i<constantCount;++i){var entryPtr=constantsPtr+32*i;var key=WebGPU.makeStringFromStringView(entryPtr+8);constants[key]=(growMemViews(),HEAPF64)[(entryPtr+24)/8]}return constants},makePipelineLayout:layoutPtr=>{if(!layoutPtr)return"auto";return WebGPU.getJsObject(layoutPtr)},makeComputeState:ptr=>{if(!ptr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(ptr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(ptr+32)/8]),Number((growMemViews(),HEAPU64)[(ptr+40)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(ptr+16)};return desc},makeComputePipelineDesc:descriptor=>{var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),compute:WebGPU.makeComputeState(descriptor+32)};return desc},makeRenderPipelineDesc:descriptor=>{function makePrimitiveState(psPtr){if(!psPtr)return undefined;return{topology:WebGPU.PrimitiveTopology[(growMemViews(),HEAP32)[(psPtr+8)/4]],stripIndexFormat:WebGPU.IndexFormat[(growMemViews(),HEAP32)[(psPtr+12)/4]],frontFace:WebGPU.FrontFace[(growMemViews(),HEAP32)[(psPtr+16)/4]],cullMode:WebGPU.CullMode[(growMemViews(),HEAP32)[(psPtr+20)/4]],unclippedDepth:!!(growMemViews(),HEAPU32)[(psPtr+24)/4]}}function makeBlendComponent(bdPtr){if(!bdPtr)return undefined;return{operation:WebGPU.BlendOperation[(growMemViews(),HEAP32)[bdPtr/4]],srcFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+4)/4]],dstFactor:WebGPU.BlendFactor[(growMemViews(),HEAP32)[(bdPtr+8)/4]]}}function makeBlendState(bsPtr){if(!bsPtr)return undefined;return{alpha:makeBlendComponent(bsPtr+12),color:makeBlendComponent(bsPtr+0)}}function makeColorState(csPtr){var format=WebGPU.TextureFormat[(growMemViews(),HEAP32)[(csPtr+8)/4]];return format?{format,blend:makeBlendState(Number((growMemViews(),HEAPU64)[(csPtr+16)/8])),writeMask:(growMemViews(),HEAPU32)[(csPtr+24)/4]}:undefined}function makeColorStates(count,csArrayPtr){var states=[];for(var i=0;i<count;++i){states.push(makeColorState(csArrayPtr+32*i))}return states}function makeStencilStateFace(ssfPtr){return{compare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[ssfPtr/4]],failOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+4)/4]],depthFailOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+8)/4]],passOp:WebGPU.StencilOperation[(growMemViews(),HEAP32)[(ssfPtr+12)/4]]}}function makeDepthStencilState(dssPtr){if(!dssPtr)return undefined;return{format:WebGPU.TextureFormat[(growMemViews(),HEAP32)[(dssPtr+8)/4]],depthWriteEnabled:!!(growMemViews(),HEAPU32)[(dssPtr+12)/4],depthCompare:WebGPU.CompareFunction[(growMemViews(),HEAP32)[(dssPtr+16)/4]],stencilFront:makeStencilStateFace(dssPtr+20),stencilBack:makeStencilStateFace(dssPtr+36),stencilReadMask:(growMemViews(),HEAPU32)[(dssPtr+52)/4],stencilWriteMask:(growMemViews(),HEAPU32)[(dssPtr+56)/4],depthBias:(growMemViews(),HEAP32)[(dssPtr+60)/4],depthBiasSlopeScale:(growMemViews(),HEAPF32)[(dssPtr+64)/4],depthBiasClamp:(growMemViews(),HEAPF32)[(dssPtr+68)/4]}}function makeVertexAttribute(vaPtr){return{format:WebGPU.VertexFormat[(growMemViews(),HEAP32)[(vaPtr+8)/4]],offset:readI53FromI64(vaPtr+16),shaderLocation:(growMemViews(),HEAPU32)[(vaPtr+24)/4]}}function makeVertexAttributes(count,vaArrayPtr){var vas=[];for(var i=0;i<count;++i){vas.push(makeVertexAttribute(vaArrayPtr+i*32))}return vas}function makeVertexBuffer(vbPtr){if(!vbPtr)return undefined;var stepMode=WebGPU.VertexStepMode[(growMemViews(),HEAP32)[(vbPtr+8)/4]];var attributeCount=Number((growMemViews(),HEAPU64)[(vbPtr+24)/8]);if(!stepMode&&!attributeCount){return null}return{arrayStride:readI53FromI64(vbPtr+16),stepMode,attributes:makeVertexAttributes(attributeCount,Number((growMemViews(),HEAPU64)[(vbPtr+32)/8]))}}function makeVertexBuffers(count,vbArrayPtr){if(!count)return undefined;var vbs=[];for(var i=0;i<count;++i){vbs.push(makeVertexBuffer(vbArrayPtr+i*40))}return vbs}function makeVertexState(viPtr){if(!viPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(viPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(viPtr+32)/8]),Number((growMemViews(),HEAPU64)[(viPtr+40)/8])),buffers:makeVertexBuffers(Number((growMemViews(),HEAPU64)[(viPtr+48)/8]),Number((growMemViews(),HEAPU64)[(viPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(viPtr+16)};return desc}function makeMultisampleState(msPtr){if(!msPtr)return undefined;return{count:(growMemViews(),HEAPU32)[(msPtr+8)/4],mask:(growMemViews(),HEAPU32)[(msPtr+12)/4],alphaToCoverageEnabled:!!(growMemViews(),HEAPU32)[(msPtr+16)/4]}}function makeFragmentState(fsPtr){if(!fsPtr)return undefined;var desc={module:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(fsPtr+8)/8])),constants:WebGPU.makePipelineConstants(Number((growMemViews(),HEAPU64)[(fsPtr+32)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+40)/8])),targets:makeColorStates(Number((growMemViews(),HEAPU64)[(fsPtr+48)/8]),Number((growMemViews(),HEAPU64)[(fsPtr+56)/8])),entryPoint:WebGPU.makeStringFromOptionalStringView(fsPtr+16)};return desc}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.makePipelineLayout(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),vertex:makeVertexState(descriptor+32),primitive:makePrimitiveState(descriptor+96),depthStencil:makeDepthStencilState(Number((growMemViews(),HEAPU64)[(descriptor+128)/8])),multisample:makeMultisampleState(descriptor+136),fragment:makeFragmentState(Number((growMemViews(),HEAPU64)[(descriptor+160)/8]))};return desc},fillLimitStruct:(limits,limitsOutPtr)=>{var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsOutPtr/8]);function setLimitValueU32(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;(growMemViews(),HEAPU32)[(basePtr+limitOffset)/4]=limitValue}function setLimitValueU64(name,basePtr,limitOffset,fallbackValue=0){var limitValue=limits[name]??fallbackValue;writeI53ToI64(basePtr+limitOffset,limitValue)}setLimitValueU32("maxTextureDimension1D",limitsOutPtr,8);setLimitValueU32("maxTextureDimension2D",limitsOutPtr,12);setLimitValueU32("maxTextureDimension3D",limitsOutPtr,16);setLimitValueU32("maxTextureArrayLayers",limitsOutPtr,20);setLimitValueU32("maxBindGroups",limitsOutPtr,24);setLimitValueU32("maxBindGroupsPlusVertexBuffers",limitsOutPtr,28);setLimitValueU32("maxBindingsPerBindGroup",limitsOutPtr,32);setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout",limitsOutPtr,36);setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout",limitsOutPtr,40);setLimitValueU32("maxSampledTexturesPerShaderStage",limitsOutPtr,44);setLimitValueU32("maxSamplersPerShaderStage",limitsOutPtr,48);setLimitValueU32("maxStorageBuffersPerShaderStage",limitsOutPtr,52);setLimitValueU32("maxStorageTexturesPerShaderStage",limitsOutPtr,56);setLimitValueU32("maxUniformBuffersPerShaderStage",limitsOutPtr,60);setLimitValueU32("minUniformBufferOffsetAlignment",limitsOutPtr,80);setLimitValueU32("minStorageBufferOffsetAlignment",limitsOutPtr,84);setLimitValueU64("maxUniformBufferBindingSize",limitsOutPtr,64);setLimitValueU64("maxStorageBufferBindingSize",limitsOutPtr,72);setLimitValueU32("maxVertexBuffers",limitsOutPtr,88);setLimitValueU64("maxBufferSize",limitsOutPtr,96);setLimitValueU32("maxVertexAttributes",limitsOutPtr,104);setLimitValueU32("maxVertexBufferArrayStride",limitsOutPtr,108);setLimitValueU32("maxInterStageShaderVariables",limitsOutPtr,112);setLimitValueU32("maxColorAttachments",limitsOutPtr,116);setLimitValueU32("maxColorAttachmentBytesPerSample",limitsOutPtr,120);setLimitValueU32("maxComputeWorkgroupStorageSize",limitsOutPtr,124);setLimitValueU32("maxComputeInvocationsPerWorkgroup",limitsOutPtr,128);setLimitValueU32("maxComputeWorkgroupSizeX",limitsOutPtr,132);setLimitValueU32("maxComputeWorkgroupSizeY",limitsOutPtr,136);setLimitValueU32("maxComputeWorkgroupSizeZ",limitsOutPtr,140);setLimitValueU32("maxComputeWorkgroupsPerDimension",limitsOutPtr,144);setLimitValueU32("maxImmediateSize",limitsOutPtr,148);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;setLimitValueU32("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24,limits.maxStorageBuffersPerShaderStage);setLimitValueU32("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20,limits.maxStorageTexturesPerShaderStage);setLimitValueU32("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28,limits.maxStorageTexturesPerShaderStage)}},fillAdapterInfoStruct:(info,infoStruct)=>{(growMemViews(),HEAPU32)[(infoStruct+88)/4]=info.subgroupMinSize;(growMemViews(),HEAPU32)[(infoStruct+92)/4]=info.subgroupMaxSize;var strs=info.vendor+info.architecture+info.device+info.description;var strPtr=stringToNewUTF8(strs);var vendorLen=lengthBytesUTF8(info.vendor);WebGPU.setStringView(infoStruct+8,strPtr,vendorLen);strPtr+=vendorLen;var architectureLen=lengthBytesUTF8(info.architecture);WebGPU.setStringView(infoStruct+24,strPtr,architectureLen);strPtr+=architectureLen;var deviceLen=lengthBytesUTF8(info.device);WebGPU.setStringView(infoStruct+40,strPtr,deviceLen);strPtr+=deviceLen;var descriptionLen=lengthBytesUTF8(info.description);WebGPU.setStringView(infoStruct+56,strPtr,descriptionLen);strPtr+=descriptionLen;(growMemViews(),HEAP32)[(infoStruct+72)/4]=2;var adapterType=info.isFallbackAdapter?3:4;(growMemViews(),HEAP32)[(infoStruct+76)/4]=adapterType;(growMemViews(),HEAPU32)[(infoStruct+80)/4]=0;(growMemViews(),HEAPU32)[(infoStruct+84)/4]=0},AddressMode:[,"clamp-to-edge","repeat","mirror-repeat"],BlendFactor:[,"zero","one","src","one-minus-src","src-alpha","one-minus-src-alpha","dst","one-minus-dst","dst-alpha","one-minus-dst-alpha","src-alpha-saturated","constant","one-minus-constant","src1","one-minus-src1","src1-alpha","one-minus-src1-alpha"],BlendOperation:[,"add","subtract","reverse-subtract","min","max"],BufferBindingType:[,,"uniform","storage","read-only-storage"],BufferMapState:[,"unmapped","pending","mapped"],CompareFunction:[,"never","less","equal","less-equal","greater","not-equal","greater-equal","always"],CompilationInfoRequestStatus:[,"success","callback-cancelled"],ComponentSwizzle:[,"0","1","r","g","b","a"],CompositeAlphaMode:[,"opaque","premultiplied","unpremultiplied","inherit"],CullMode:[,"none","front","back"],ErrorFilter:[,"validation","out-of-memory","internal"],FeatureLevel:[,"compatibility","core"],FeatureName:{1:"core-features-and-limits",2:"depth-clip-control",3:"depth32float-stencil8",4:"texture-compression-bc",5:"texture-compression-bc-sliced-3d",6:"texture-compression-etc2",7:"texture-compression-astc",8:"texture-compression-astc-sliced-3d",9:"timestamp-query",10:"indirect-first-instance",11:"shader-f16",12:"rg11b10ufloat-renderable",13:"bgra8unorm-storage",14:"float32-filterable",15:"float32-blendable",16:"clip-distances",17:"dual-source-blending",18:"subgroups",19:"texture-formats-tier1",20:"texture-formats-tier2",21:"primitive-index",22:"texture-component-swizzle",327692:"chromium-experimental-unorm16-texture-formats",327729:"chromium-experimental-multi-draw-indirect"},FilterMode:[,"nearest","linear"],FrontFace:[,"ccw","cw"],IndexFormat:[,"uint16","uint32"],InstanceFeatureName:[,"timed-wait-any","shader-source-spirv","multiple-devices-per-adapter"],LoadOp:[,"load","clear"],MipmapFilterMode:[,"nearest","linear"],OptionalBool:["false","true"],PowerPreference:[,"low-power","high-performance"],PredefinedColorSpace:[,"srgb","display-p3"],PrimitiveTopology:[,"point-list","line-list","line-strip","triangle-list","triangle-strip"],QueryType:[,"occlusion","timestamp"],SamplerBindingType:[,,"filtering","non-filtering","comparison"],Status:[,"success","error"],StencilOperation:[,"keep","zero","replace","invert","increment-clamp","decrement-clamp","increment-wrap","decrement-wrap"],StorageTextureAccess:[,,"write-only","read-only","read-write"],StoreOp:[,"store","discard"],SurfaceGetCurrentTextureStatus:[,"success-optimal","success-suboptimal","timeout","outdated","lost","error"],TextureAspect:[,"all","stencil-only","depth-only"],TextureDimension:[,"1d","2d","3d"],TextureFormat:[,"r8unorm","r8snorm","r8uint","r8sint","r16unorm","r16snorm","r16uint","r16sint","r16float","rg8unorm","rg8snorm","rg8uint","rg8sint","r32float","r32uint","r32sint","rg16unorm","rg16snorm","rg16uint","rg16sint","rg16float","rgba8unorm","rgba8unorm-srgb","rgba8snorm","rgba8uint","rgba8sint","bgra8unorm","bgra8unorm-srgb","rgb10a2uint","rgb10a2unorm","rg11b10ufloat","rgb9e5ufloat","rg32float","rg32uint","rg32sint","rgba16unorm","rgba16snorm","rgba16uint","rgba16sint","rgba16float","rgba32float","rgba32uint","rgba32sint","stencil8","depth16unorm","depth24plus","depth24plus-stencil8","depth32float","depth32float-stencil8","bc1-rgba-unorm","bc1-rgba-unorm-srgb","bc2-rgba-unorm","bc2-rgba-unorm-srgb","bc3-rgba-unorm","bc3-rgba-unorm-srgb","bc4-r-unorm","bc4-r-snorm","bc5-rg-unorm","bc5-rg-snorm","bc6h-rgb-ufloat","bc6h-rgb-float","bc7-rgba-unorm","bc7-rgba-unorm-srgb","etc2-rgb8unorm","etc2-rgb8unorm-srgb","etc2-rgb8a1unorm","etc2-rgb8a1unorm-srgb","etc2-rgba8unorm","etc2-rgba8unorm-srgb","eac-r11unorm","eac-r11snorm","eac-rg11unorm","eac-rg11snorm","astc-4x4-unorm","astc-4x4-unorm-srgb","astc-5x4-unorm","astc-5x4-unorm-srgb","astc-5x5-unorm","astc-5x5-unorm-srgb","astc-6x5-unorm","astc-6x5-unorm-srgb","astc-6x6-unorm","astc-6x6-unorm-srgb","astc-8x5-unorm","astc-8x5-unorm-srgb","astc-8x6-unorm","astc-8x6-unorm-srgb","astc-8x8-unorm","astc-8x8-unorm-srgb","astc-10x5-unorm","astc-10x5-unorm-srgb","astc-10x6-unorm","astc-10x6-unorm-srgb","astc-10x8-unorm","astc-10x8-unorm-srgb","astc-10x10-unorm","astc-10x10-unorm-srgb","astc-12x10-unorm","astc-12x10-unorm-srgb","astc-12x12-unorm","astc-12x12-unorm-srgb"],TextureSampleType:[,,"float","unfilterable-float","depth","sint","uint"],TextureViewDimension:[,"1d","2d","2d-array","cube","cube-array","3d"],ToneMappingMode:[,"standard","extended"],VertexFormat:[,"uint8","uint8x2","uint8x4","sint8","sint8x2","sint8x4","unorm8","unorm8x2","unorm8x4","snorm8","snorm8x2","snorm8x4","uint16","uint16x2","uint16x4","sint16","sint16x2","sint16x4","unorm16","unorm16x2","unorm16x4","snorm16","snorm16x2","snorm16x4","float16","float16x2","float16x4","float32","float32x2","float32x3","float32x4","uint32","uint32x2","uint32x3","uint32x4","sint32","sint32x2","sint32x3","sint32x4","unorm10-10-10-2","unorm8x4-bgra"],VertexStepMode:[,"vertex","instance"],WGSLLanguageFeatureName:[,"readonly_and_readwrite_storage_textures","packed_4x8_integer_dot_product","unrestricted_pointer_parameters","pointer_composite_access","uniform_buffer_standard_layout","subgroup_id","texture_and_sampler_let","subgroup_uniformity","texture_formats_tier1"]};var emwgpuStringToInt_DeviceLostReason={undefined:1,unknown:1,destroyed:2};var runtimeKeepalivePop=()=>{runtimeKeepaliveCounter-=1};function _emwgpuAdapterRequestDevice(adapterPtr,futureId,deviceLostFutureId,devicePtr,queuePtr,descriptor){adapterPtr=bigintToI53Checked(adapterPtr);futureId=bigintToI53Checked(futureId);deviceLostFutureId=bigintToI53Checked(deviceLostFutureId);devicePtr=bigintToI53Checked(devicePtr);queuePtr=bigintToI53Checked(queuePtr);descriptor=bigintToI53Checked(descriptor);var adapter=WebGPU.getJsObject(adapterPtr);var desc={};if(descriptor){var requiredFeatureCount=Number((growMemViews(),HEAPU64)[(descriptor+24)/8]);if(requiredFeatureCount){var requiredFeaturesPtr=Number((growMemViews(),HEAPU64)[(descriptor+32)/8]);desc["requiredFeatures"]=Array.from((growMemViews(),HEAPU32).subarray(requiredFeaturesPtr/4,(requiredFeaturesPtr+requiredFeatureCount*4)/4),feature=>WebGPU.FeatureName[feature])}var limitsPtr=Number((growMemViews(),HEAPU64)[(descriptor+40)/8]);if(limitsPtr){var nextInChainPtr=Number((growMemViews(),HEAPU64)[limitsPtr/8]);var requiredLimits={};function setLimitU32IfDefined(name,basePtr,limitOffset,ignoreIfZero=false){var ptr=basePtr+limitOffset;var value=(growMemViews(),HEAPU32)[ptr/4];if(value!=4294967295&&(!ignoreIfZero||value!=0)){requiredLimits[name]=value}}function setLimitU64IfDefined(name,basePtr,limitOffset){var ptr=basePtr+limitOffset;var limitPart1=(growMemViews(),HEAPU32)[ptr/4];var limitPart2=(growMemViews(),HEAPU32)[(ptr+4)/4];if(limitPart1!=4294967295||limitPart2!=4294967295){requiredLimits[name]=readI53FromI64(ptr)}}setLimitU32IfDefined("maxTextureDimension1D",limitsPtr,8);setLimitU32IfDefined("maxTextureDimension2D",limitsPtr,12);setLimitU32IfDefined("maxTextureDimension3D",limitsPtr,16);setLimitU32IfDefined("maxTextureArrayLayers",limitsPtr,20);setLimitU32IfDefined("maxBindGroups",limitsPtr,24);setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers",limitsPtr,28);setLimitU32IfDefined("maxBindingsPerBindGroup",limitsPtr,32);setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout",limitsPtr,36);setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout",limitsPtr,40);setLimitU32IfDefined("maxSampledTexturesPerShaderStage",limitsPtr,44);setLimitU32IfDefined("maxSamplersPerShaderStage",limitsPtr,48);setLimitU32IfDefined("maxStorageBuffersPerShaderStage",limitsPtr,52);setLimitU32IfDefined("maxStorageTexturesPerShaderStage",limitsPtr,56);setLimitU32IfDefined("maxUniformBuffersPerShaderStage",limitsPtr,60);setLimitU32IfDefined("minUniformBufferOffsetAlignment",limitsPtr,80);setLimitU32IfDefined("minStorageBufferOffsetAlignment",limitsPtr,84);setLimitU64IfDefined("maxUniformBufferBindingSize",limitsPtr,64);setLimitU64IfDefined("maxStorageBufferBindingSize",limitsPtr,72);setLimitU32IfDefined("maxVertexBuffers",limitsPtr,88);setLimitU64IfDefined("maxBufferSize",limitsPtr,96);setLimitU32IfDefined("maxVertexAttributes",limitsPtr,104);setLimitU32IfDefined("maxVertexBufferArrayStride",limitsPtr,108);setLimitU32IfDefined("maxInterStageShaderVariables",limitsPtr,112);setLimitU32IfDefined("maxColorAttachments",limitsPtr,116);setLimitU32IfDefined("maxColorAttachmentBytesPerSample",limitsPtr,120);setLimitU32IfDefined("maxComputeWorkgroupStorageSize",limitsPtr,124);setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup",limitsPtr,128);setLimitU32IfDefined("maxComputeWorkgroupSizeX",limitsPtr,132);setLimitU32IfDefined("maxComputeWorkgroupSizeY",limitsPtr,136);setLimitU32IfDefined("maxComputeWorkgroupSizeZ",limitsPtr,140);setLimitU32IfDefined("maxComputeWorkgroupsPerDimension",limitsPtr,144);setLimitU32IfDefined("maxImmediateSize",limitsPtr,148,true);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var compatibilityModeLimitsPtr=nextInChainPtr;if("maxStorageBuffersInVertexStage"in GPUSupportedLimits.prototype){setLimitU32IfDefined("maxStorageBuffersInVertexStage",compatibilityModeLimitsPtr,16);setLimitU32IfDefined("maxStorageTexturesInVertexStage",compatibilityModeLimitsPtr,20);setLimitU32IfDefined("maxStorageBuffersInFragmentStage",compatibilityModeLimitsPtr,24);setLimitU32IfDefined("maxStorageTexturesInFragmentStage",compatibilityModeLimitsPtr,28)}}desc["requiredLimits"]=requiredLimits}var defaultQueuePtr=Number((growMemViews(),HEAPU64)[(descriptor+48)/8]);if(defaultQueuePtr){var defaultQueueDesc={label:WebGPU.makeStringFromOptionalStringView(defaultQueuePtr+8)};desc["defaultQueue"]=defaultQueueDesc}desc["label"]=WebGPU.makeStringFromOptionalStringView(descriptor+8)}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,adapter.requestDevice(desc).then(device=>{runtimeKeepalivePop();callUserCallback(()=>{WebGPU.Internals.jsObjectInsert(queuePtr,device.queue);WebGPU.Internals.jsObjectInsert(devicePtr,device);devicePtr=BigInt(devicePtr);WebGPU.Internals.futureInsert(deviceLostFutureId,device.lost.then(info=>{callUserCallback(()=>{device.onuncapturederror=ev=>{};var sp=stackSave();var messagePtr=stringToUTF8OnStack(info.message);_emwgpuOnDeviceLostCompleted(deviceLostFutureId,emwgpuStringToInt_DeviceLostReason[info.reason],BigInt(messagePtr));stackRestore(sp)})}));device.onuncapturederror=ev=>{var type=5;if(ev.error instanceof GPUValidationError)type=2;else if(ev.error instanceof GPUOutOfMemoryError)type=3;else if(ev.error instanceof GPUInternalError)type=4;var sp=stackSave();var messagePtr=stringToUTF8OnStack(ev.error.message);_emwgpuOnUncapturedError(BigInt(devicePtr),type,BigInt(messagePtr));stackRestore(sp)};_emwgpuOnRequestDeviceCompleted(futureId,1,BigInt(devicePtr),0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestDeviceCompleted(futureId,3,BigInt(devicePtr),BigInt(messagePtr));if(deviceLostFutureId){_emwgpuOnDeviceLostCompleted(deviceLostFutureId,4,BigInt(messagePtr))}stackRestore(sp)})}))}function _emwgpuBufferDestroy(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(onUnmap){for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]}buffer.destroy()}var warnOnce=text=>{warnOnce.shown||={};if(!warnOnce.shown[text]){warnOnce.shown[text]=1;if(ENVIRONMENT_IS_NODE)text="warning: "+text;err(text)}};var _emwgpuBufferGetConstMappedRange=function(bufferPtr,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);if(size==-1)size=undefined;var mapped;try{mapped=buffer.getMappedRange(offset,size)}catch(ex){return 0n}var data=_memalign(16,mapped.byteLength);(growMemViews(),HEAPU8).set(new Uint8Array(mapped),data);WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(()=>_free(data));return data})();return BigInt(ret)};var _emwgpuBufferMapAsync=function(bufferPtr,futureId,mode,offset,size){bufferPtr=bigintToI53Checked(bufferPtr);futureId=bigintToI53Checked(futureId);mode=bigintToI53Checked(mode);offset=bigintToI53Checked(offset);size=bigintToI53Checked(size);var buffer=WebGPU.getJsObject(bufferPtr);WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[];if(size==-1)size=undefined;runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,buffer.mapAsync(mode,offset,size).then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnMapAsyncCompleted(futureId,1,0n)})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);var status=ex.name==="AbortError"?4:ex.name==="OperationError"?3:0;_emwgpuOnMapAsyncCompleted(futureId,status,BigInt(messagePtr));delete WebGPU.Internals.bufferOnUnmaps[bufferPtr]})}))};function _emwgpuBufferUnmap(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var buffer=WebGPU.getJsObject(bufferPtr);var onUnmap=WebGPU.Internals.bufferOnUnmaps[bufferPtr];if(!onUnmap){return}for(var i=0;i<onUnmap.length;++i){onUnmap[i]()}delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];buffer.unmap()}function _emwgpuDelete(ptr){ptr=bigintToI53Checked(ptr);delete WebGPU.Internals.jsObjects[ptr]}function _emwgpuDeviceCreateBuffer(devicePtr,descriptor,bufferPtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);bufferPtr=bigintToI53Checked(bufferPtr);var mappedAtCreation=!!(growMemViews(),HEAPU32)[(descriptor+40)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),usage:(growMemViews(),HEAPU32)[(descriptor+24)/4],size:readI53FromI64(descriptor+32),mappedAtCreation};var device=WebGPU.getJsObject(devicePtr);var buffer;try{buffer=device.createBuffer(desc)}catch(ex){return false}WebGPU.Internals.jsObjectInsert(bufferPtr,buffer);if(mappedAtCreation){WebGPU.Internals.bufferOnUnmaps[bufferPtr]=[]}return true}function _emwgpuDeviceCreateShaderModule(devicePtr,descriptor,shaderModulePtr){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);shaderModulePtr=bigintToI53Checked(shaderModulePtr);var nextInChainPtr=Number((growMemViews(),HEAPU64)[descriptor/8]);var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),code:""};switch(sType){case 2:{desc["code"]=WebGPU.makeStringFromStringView(nextInChainPtr+16);break}}var device=WebGPU.getJsObject(devicePtr);WebGPU.Internals.jsObjectInsert(shaderModulePtr,device.createShaderModule(desc))}var _emwgpuDeviceDestroy=devicePtr=>{const device=WebGPU.getJsObject(devicePtr);device.onuncapturederror=null;device.destroy()};function _emwgpuInstanceRequestAdapter(instancePtr,futureId,options,adapterPtr){instancePtr=bigintToI53Checked(instancePtr);futureId=bigintToI53Checked(futureId);options=bigintToI53Checked(options);adapterPtr=bigintToI53Checked(adapterPtr);var opts;if(options){opts={featureLevel:WebGPU.FeatureLevel[(growMemViews(),HEAP32)[(options+8)/4]],powerPreference:WebGPU.PowerPreference[(growMemViews(),HEAP32)[(options+12)/4]],forceFallbackAdapter:!!(growMemViews(),HEAPU32)[(options+16)/4]};var nextInChainPtr=Number((growMemViews(),HEAPU64)[options/8]);if(nextInChainPtr!==0){var sType=(growMemViews(),HEAP32)[(nextInChainPtr+8)/4];var webxrOptions=nextInChainPtr;opts.xrCompatible=!!(growMemViews(),HEAPU32)[(webxrOptions+16)/4]}}if(!("gpu"in navigator)){var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp);return}runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,navigator.gpu.requestAdapter(opts).then(adapter=>{runtimeKeepalivePop();callUserCallback(()=>{if(adapter){WebGPU.Internals.jsObjectInsert(adapterPtr,adapter);_emwgpuOnRequestAdapterCompleted(futureId,1,BigInt(adapterPtr),0n)}else{var sp=stackSave();var messagePtr=stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");_emwgpuOnRequestAdapterCompleted(futureId,3,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)}})},ex=>{runtimeKeepalivePop();callUserCallback(()=>{var sp=stackSave();var messagePtr=stringToUTF8OnStack(ex.message);_emwgpuOnRequestAdapterCompleted(futureId,4,BigInt(adapterPtr),BigInt(messagePtr));stackRestore(sp)})}))}var _emwgpuQueueOnSubmittedWorkDone=function(queuePtr,futureId){queuePtr=bigintToI53Checked(queuePtr);futureId=bigintToI53Checked(futureId);var queue=WebGPU.getJsObject(queuePtr);runtimeKeepalivePush();WebGPU.Internals.futureInsert(futureId,queue.onSubmittedWorkDone().then(()=>{runtimeKeepalivePop();callUserCallback(()=>{_emwgpuOnWorkDoneCompleted(futureId,1)})}))};var _emwgpuWaitAny=function(futurePtr,futureCount,timeoutMSPtr){futurePtr=bigintToI53Checked(futurePtr);futureCount=bigintToI53Checked(futureCount);timeoutMSPtr=bigintToI53Checked(timeoutMSPtr);return Asyncify.handleAsync(async()=>{var promises=[];if(timeoutMSPtr){var timeoutMS=(growMemViews(),HEAP32)[timeoutMSPtr/4];promises.length=futureCount+1;promises[futureCount]=new Promise(resolve=>setTimeout(resolve,timeoutMS,0))}else{promises.length=futureCount}for(var i=0;i<futureCount;++i){var futureId=readI53FromI64(futurePtr+i*8);if(!(futureId in WebGPU.Internals.futures)){return futureId}promises[i]=WebGPU.Internals.futures[futureId]}const firstResolvedFuture=await Promise.race(promises);delete WebGPU.Internals.futures[firstResolvedFuture];return firstResolvedFuture})};_emwgpuWaitAny.isAsync=true;var ENV={};var getExecutableName=()=>thisProgram||"./this.program";var getEnvStrings=()=>{if(!getEnvStrings.strings){var lang=(typeof navigator=="object"&&navigator.language||"C").replace("-","_")+".UTF-8";var env={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:lang,_:getExecutableName()};for(var x in ENV){if(ENV[x]===undefined)delete env[x];else env[x]=ENV[x]}var strings=[];for(var x in env){strings.push(`${x}=${env[x]}`)}getEnvStrings.strings=strings}return getEnvStrings.strings};function _environ_get(__environ,environ_buf){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(14,0,1,__environ,environ_buf);__environ=bigintToI53Checked(__environ);environ_buf=bigintToI53Checked(environ_buf);var bufSize=0;var envp=0;for(var string of getEnvStrings()){var ptr=environ_buf+bufSize;(growMemViews(),HEAPU64)[(__environ+envp)/8]=BigInt(ptr);bufSize+=stringToUTF8(string,ptr,Infinity)+1;envp+=8}return 0}function _environ_sizes_get(penviron_count,penviron_buf_size){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(15,0,1,penviron_count,penviron_buf_size);penviron_count=bigintToI53Checked(penviron_count);penviron_buf_size=bigintToI53Checked(penviron_buf_size);var strings=getEnvStrings();(growMemViews(),HEAPU64)[penviron_count/8]=BigInt(strings.length);var bufSize=0;for(var string of strings){bufSize+=lengthBytesUTF8(string)+1}(growMemViews(),HEAPU64)[penviron_buf_size/8]=BigInt(bufSize);return 0}function _fd_close(fd){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(16,0,1,fd);try{var stream=SYSCALLS.getStreamFromFD(fd);FS.close(stream);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doReadv=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.read(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len)break;if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_read(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(17,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doReadv(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _fd_seek(fd,offset,whence,newOffset){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(18,0,1,fd,offset,whence,newOffset);offset=bigintToI53Checked(offset);newOffset=bigintToI53Checked(newOffset);try{if(isNaN(offset))return 61;var stream=SYSCALLS.getStreamFromFD(fd);FS.llseek(stream,offset,whence);(growMemViews(),HEAP64)[newOffset/8]=BigInt(stream.position);if(stream.getdents&&offset===0&&whence===0)stream.getdents=null;return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}var doWritev=(stream,iov,iovcnt,offset)=>{var ret=0;for(var i=0;i<iovcnt;i++){var ptr=Number((growMemViews(),HEAPU64)[iov/8]);var len=Number((growMemViews(),HEAPU64)[(iov+8)/8]);iov+=16;var curr=FS.write(stream,(growMemViews(),HEAP8),ptr,len,offset);if(curr<0)return-1;ret+=curr;if(curr<len){break}if(typeof offset!="undefined"){offset+=curr}}return ret};function _fd_write(fd,iov,iovcnt,pnum){if(ENVIRONMENT_IS_PTHREAD)return proxyToMainThread(19,0,1,fd,iov,iovcnt,pnum);iov=bigintToI53Checked(iov);iovcnt=bigintToI53Checked(iovcnt);pnum=bigintToI53Checked(pnum);try{var stream=SYSCALLS.getStreamFromFD(fd);var num=doWritev(stream,iov,iovcnt);(growMemViews(),HEAPU64)[pnum/8]=BigInt(num);return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _random_get(buffer,size){buffer=bigintToI53Checked(buffer);size=bigintToI53Checked(size);try{randomFill((growMemViews(),HEAPU8).subarray(buffer,buffer+size));return 0}catch(e){if(typeof FS=="undefined"||!(e.name==="ErrnoError"))throw e;return e.errno}}function _wgpuAdapterGetInfo(adapterPtr,info){adapterPtr=bigintToI53Checked(adapterPtr);info=bigintToI53Checked(info);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillAdapterInfoStruct(adapter.info,info);return 1}function _wgpuAdapterGetLimits(adapterPtr,limitsOutPtr){adapterPtr=bigintToI53Checked(adapterPtr);limitsOutPtr=bigintToI53Checked(limitsOutPtr);var adapter=WebGPU.getJsObject(adapterPtr);WebGPU.fillLimitStruct(adapter.limits,limitsOutPtr);return 1}function _wgpuAdapterHasFeature(adapterPtr,featureEnumValue){adapterPtr=bigintToI53Checked(adapterPtr);var adapter=WebGPU.getJsObject(adapterPtr);return adapter.features.has(WebGPU.FeatureName[featureEnumValue])}var _wgpuBufferGetSize=function(bufferPtr){bufferPtr=bigintToI53Checked(bufferPtr);var ret=(()=>{var buffer=WebGPU.getJsObject(bufferPtr);return buffer.size})();return BigInt(ret)};var _wgpuCommandEncoderBeginComputePass=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),timestampWrites:WebGPU.makePassTimestampWrites(Number((growMemViews(),HEAPU64)[(descriptor+24)/8]))}}var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateComputePassEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.beginComputePass(desc));return ptr})();return BigInt(ret)};function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr,srcPtr,srcOffset,dstPtr,dstOffset,size){encoderPtr=bigintToI53Checked(encoderPtr);srcPtr=bigintToI53Checked(srcPtr);srcOffset=bigintToI53Checked(srcOffset);dstPtr=bigintToI53Checked(dstPtr);dstOffset=bigintToI53Checked(dstOffset);size=bigintToI53Checked(size);var commandEncoder=WebGPU.getJsObject(encoderPtr);var src=WebGPU.getJsObject(srcPtr);var dst=WebGPU.getJsObject(dstPtr);commandEncoder.copyBufferToBuffer(src,srcOffset,dst,dstOffset,size)}var _wgpuCommandEncoderFinish=function(encoderPtr,descriptor){encoderPtr=bigintToI53Checked(encoderPtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var commandEncoder=WebGPU.getJsObject(encoderPtr);var ptr=_emwgpuCreateCommandBuffer(0n);WebGPU.Internals.jsObjectInsert(ptr,commandEncoder.finish());return ptr})();return BigInt(ret)};function _wgpuComputePassEncoderDispatchWorkgroups(passPtr,x,y,z){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.dispatchWorkgroups(x,y,z)}function _wgpuComputePassEncoderEnd(passPtr){passPtr=bigintToI53Checked(passPtr);var pass=WebGPU.getJsObject(passPtr);pass.end()}function _wgpuComputePassEncoderSetBindGroup(passPtr,groupIndex,groupPtr,dynamicOffsetCount,dynamicOffsetsPtr){passPtr=bigintToI53Checked(passPtr);groupPtr=bigintToI53Checked(groupPtr);dynamicOffsetCount=bigintToI53Checked(dynamicOffsetCount);dynamicOffsetsPtr=bigintToI53Checked(dynamicOffsetsPtr);var pass=WebGPU.getJsObject(passPtr);var group=WebGPU.getJsObject(groupPtr);if(dynamicOffsetCount==0){pass.setBindGroup(groupIndex,group)}else{pass.setBindGroup(groupIndex,group,(growMemViews(),HEAPU32),dynamicOffsetsPtr/4,dynamicOffsetCount)}}function _wgpuComputePassEncoderSetPipeline(passPtr,pipelinePtr){passPtr=bigintToI53Checked(passPtr);pipelinePtr=bigintToI53Checked(pipelinePtr);var pass=WebGPU.getJsObject(passPtr);var pipeline=WebGPU.getJsObject(pipelinePtr);pass.setPipeline(pipeline)}var _wgpuComputePipelineGetBindGroupLayout=function(pipelinePtr,groupIndex){pipelinePtr=bigintToI53Checked(pipelinePtr);var ret=(()=>{var pipeline=WebGPU.getJsObject(pipelinePtr);var ptr=_emwgpuCreateBindGroupLayout(0n);WebGPU.Internals.jsObjectInsert(ptr,pipeline.getBindGroupLayout(groupIndex));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateBindGroup=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{function makeEntry(entryPtr){var bufferPtr=Number((growMemViews(),HEAPU64)[(entryPtr+16)/8]);var samplerPtr=Number((growMemViews(),HEAPU64)[(entryPtr+40)/8]);var textureViewPtr=Number((growMemViews(),HEAPU64)[(entryPtr+48)/8]);var externalTexturePtr=0;WebGPU.iterateExtensions(entryPtr,{327681:ptr=>{externalTexturePtr=Number((growMemViews(),HEAPU64)[(ptr+16)/8])}});var resource;if(bufferPtr){var size=readI53FromI64(entryPtr+32);if(size==-1)size=undefined;resource={buffer:WebGPU.getJsObject(bufferPtr),offset:readI53FromI64(entryPtr+24),size}}else{resource=WebGPU.getJsObject(samplerPtr||textureViewPtr||externalTexturePtr)}return{binding:(growMemViews(),HEAPU32)[(entryPtr+8)/4],resource}}function makeEntries(count,entriesPtrs){var entries=[];for(var i=0;i<count;++i){entries.push(makeEntry(entriesPtrs+56*i))}return entries}var desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8),layout:WebGPU.getJsObject(Number((growMemViews(),HEAPU64)[(descriptor+24)/8])),entries:makeEntries(Number((growMemViews(),HEAPU64)[(descriptor+32)/8]),Number((growMemViews(),HEAPU64)[(descriptor+40)/8]))};var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateBindGroup(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createBindGroup(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateCommandEncoder=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc;if(descriptor){desc={label:WebGPU.makeStringFromOptionalStringView(descriptor+8)}}var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateCommandEncoder(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createCommandEncoder(desc));return ptr})();return BigInt(ret)};var _wgpuDeviceCreateComputePipeline=function(devicePtr,descriptor){devicePtr=bigintToI53Checked(devicePtr);descriptor=bigintToI53Checked(descriptor);var ret=(()=>{var desc=WebGPU.makeComputePipelineDesc(descriptor);var device=WebGPU.getJsObject(devicePtr);var ptr=_emwgpuCreateComputePipeline(0n);WebGPU.Internals.jsObjectInsert(ptr,device.createComputePipeline(desc));return ptr})();return BigInt(ret)};function _wgpuInstanceHasWGSLLanguageFeature(instance,featureEnumValue){instance=bigintToI53Checked(instance);if(!("wgslLanguageFeatures"in navigator.gpu)){return false}return navigator.gpu.wgslLanguageFeatures.has(WebGPU.WGSLLanguageFeatureName[featureEnumValue])}var _wgpuQueueSubmit=function(queuePtr,commandCount,commands){queuePtr=bigintToI53Checked(queuePtr);commandCount=bigintToI53Checked(commandCount);commands=bigintToI53Checked(commands);var queue=WebGPU.getJsObject(queuePtr);var cmds=Array.from((growMemViews(),HEAP64).subarray(commands/8,(commands+commandCount*8)/8),id=>WebGPU.getJsObject(id));queue.submit(cmds)};function _wgpuQueueWriteBuffer(queuePtr,bufferPtr,bufferOffset,data,size){queuePtr=bigintToI53Checked(queuePtr);bufferPtr=bigintToI53Checked(bufferPtr);bufferOffset=bigintToI53Checked(bufferOffset);data=bigintToI53Checked(data);size=bigintToI53Checked(size);var queue=WebGPU.getJsObject(queuePtr);var buffer=WebGPU.getJsObject(bufferPtr);var subarray=(growMemViews(),HEAPU8).subarray(data,data+size);queue.writeBuffer(buffer,bufferOffset,subarray,0,size)}var Asyncify={instrumentWasmImports(imports){var importPattern=/^(invoke_.*|__asyncjs__.*)$/;for(let[x,original]of Object.entries(imports)){if(typeof original=="function"){let isAsyncifyImport=original.isAsync||importPattern.test(x);if(isAsyncifyImport){imports[x]=original=new WebAssembly.Suspending(original)}}}},instrumentFunction(original){var wrapper=(...args)=>original(...args);return wrapper},instrumentWasmExports(exports){var exportPattern=/^(wllama_start|wllama_action|main|__main_argc_argv)$/;Asyncify.asyncExports=new Set;var ret={};for(let[x,original]of Object.entries(exports)){if(typeof original=="function"){let isAsyncifyExport=exportPattern.test(x);if(isAsyncifyExport){Asyncify.asyncExports.add(original);original=Asyncify.makeAsyncFunction(original)}var wrapper=Asyncify.instrumentFunction(original);ret[x]=wrapper}else{ret[x]=original}}return ret},asyncExports:null,isAsyncExport(func){return Asyncify.asyncExports?.has(func)},handleAsync:async startAsync=>{runtimeKeepalivePush();try{return await startAsync()}finally{runtimeKeepalivePop()}},handleSleep:startAsync=>Asyncify.handleAsync(()=>new Promise(startAsync)),makeAsyncFunction(original){return WebAssembly.promising(original)}};var getCFunc=ident=>{var func=Module["_"+ident];return func};var writeArrayToMemory=(array,buffer)=>{(growMemViews(),HEAP8).set(array,buffer)};var ccall=(ident,returnType,argTypes,args,opts)=>{var toC={pointer:p=>BigInt(p),string:str=>{var ret=0;if(str!==null&&str!==undefined&&str!==0){ret=stringToUTF8OnStack(str)}return BigInt(ret)},array:arr=>{var ret=stackAlloc(arr.length);writeArrayToMemory(arr,ret);return BigInt(ret)}};function convertReturnValue(ret){if(returnType==="string"){return UTF8ToString(Number(ret))}if(returnType==="pointer")return Number(ret);if(returnType==="boolean")return Boolean(ret);return ret}var func=getCFunc(ident);var cArgs=[];var stack=0;if(args){for(var i=0;i<args.length;i++){var converter=toC[argTypes[i]];if(converter){if(stack===0)stack=stackSave();cArgs[i]=converter(args[i])}else{cArgs[i]=args[i]}}}var ret=func(...cArgs);function onDone(ret){if(stack!==0)stackRestore(stack);return convertReturnValue(ret)}var asyncMode=opts?.async;if(asyncMode)return ret.then(onDone);ret=onDone(ret);return ret};var cwrap=(ident,returnType,argTypes,opts)=>{var numericArgs=!argTypes||argTypes.every(type=>type==="number"||type==="boolean");var numericRet=returnType!=="string";if(numericRet&&numericArgs&&!opts){return getCFunc(ident)}return(...args)=>ccall(ident,returnType,argTypes,args,opts)};var FS_createPath=(...args)=>FS.createPath(...args);var FS_unlink=(...args)=>FS.unlink(...args);var FS_createLazyFile=(...args)=>FS.createLazyFile(...args);var FS_createDevice=(...args)=>FS.createDevice(...args);PThread.init();FS.createPreloadedFile=FS_createPreloadedFile;FS.preloadFile=FS_preloadFile;FS.staticInit();{initMemory();if(Module["noExitRuntime"])noExitRuntime=Module["noExitRuntime"];if(Module["preloadPlugins"])preloadPlugins=Module["preloadPlugins"];if(Module["print"])out=Module["print"];if(Module["printErr"])err=Module["printErr"];if(Module["wasmBinary"])wasmBinary=Module["wasmBinary"];if(Module["arguments"])arguments_=Module["arguments"];if(Module["thisProgram"])thisProgram=Module["thisProgram"];if(Module["preInit"]){if(typeof Module["preInit"]=="function")Module["preInit"]=[Module["preInit"]];while(Module["preInit"].length>0){Module["preInit"].shift()()}}}Module["ENV"]=ENV;Module["mmapAlloc"]=mmapAlloc;Module["wasmMemory"]=wasmMemory;Module["addRunDependency"]=addRunDependency;Module["removeRunDependency"]=removeRunDependency;Module["ccall"]=ccall;Module["cwrap"]=cwrap;Module["FS_preloadFile"]=FS_preloadFile;Module["FS_unlink"]=FS_unlink;Module["FS_createPath"]=FS_createPath;Module["FS_createDevice"]=FS_createDevice;Module["FS"]=FS;Module["FS_createDataFile"]=FS_createDataFile;Module["FS_createLazyFile"]=FS_createLazyFile;Module["MEMFS"]=MEMFS;var proxiedFunctionTable=[_proc_exit,exitOnMainThread,pthreadCreateProxied,___syscall_fcntl64,___syscall_fstat64,___syscall_getcwd,___syscall_getdents64,___syscall_ioctl,___syscall_lstat64,___syscall_newfstatat,___syscall_openat,___syscall_stat64,__mmap_js,__munmap_js,_environ_get,_environ_sizes_get,_fd_close,_fd_read,_fd_seek,_fd_write];function __asyncjs__js_file_read(path_ptr,offset,req_size,out_ptr){return Asyncify.handleAsync(async()=>await _wllama_js_file_read(UTF8ToString(Number(path_ptr)),Number(offset),Number(req_size),Number(out_ptr)))}__asyncjs__js_file_read.sig="jjjjj";var _wllama_malloc,_wllama_start,_wllama_action,_wllama_exit,_wllama_debug,_main,_malloc,_free,_emwgpuCreateBindGroup,_emwgpuCreateBindGroupLayout,_emwgpuCreateCommandBuffer,_emwgpuCreateCommandEncoder,_emwgpuCreateComputePassEncoder,_emwgpuCreateComputePipeline,_emwgpuCreateExternalTexture,_emwgpuCreatePipelineLayout,_emwgpuCreateQuerySet,_emwgpuCreateRenderBundle,_emwgpuCreateRenderBundleEncoder,_emwgpuCreateRenderPassEncoder,_emwgpuCreateRenderPipeline,_emwgpuCreateSampler,_emwgpuCreateSurface,_emwgpuCreateTexture,_emwgpuCreateTextureView,_emwgpuCreateAdapter,_emwgpuCreateBuffer,_emwgpuCreateDevice,_emwgpuCreateQueue,_emwgpuCreateShaderModule,_emwgpuOnDeviceLostCompleted,_emwgpuOnMapAsyncCompleted,_emwgpuOnRequestAdapterCompleted,_emwgpuOnRequestDeviceCompleted,_emwgpuOnWorkDoneCompleted,_emwgpuOnUncapturedError,__emscripten_tls_init,_pthread_self,_emscripten_builtin_memalign,__emscripten_thread_init,__emscripten_thread_crashed,__emscripten_run_js_on_main_thread,__emscripten_thread_free_data,__emscripten_thread_exit,__emscripten_check_mailbox,_memalign,___trap,_emscripten_stack_set_limits,__emscripten_stack_restore,__emscripten_stack_alloc,_emscripten_stack_get_current,__indirect_function_table,wasmTable;function assignWasmExports(wasmExports){_wllama_malloc=Module["_wllama_malloc"]=wasmExports["wllama_malloc"];_wllama_start=Module["_wllama_start"]=wasmExports["wllama_start"];_wllama_action=Module["_wllama_action"]=wasmExports["wllama_action"];_wllama_exit=Module["_wllama_exit"]=wasmExports["wllama_exit"];_wllama_debug=Module["_wllama_debug"]=wasmExports["wllama_debug"];_main=Module["_main"]=wasmExports["main"];_malloc=wasmExports["malloc"];_free=wasmExports["free"];_emwgpuCreateBindGroup=wasmExports["emwgpuCreateBindGroup"];_emwgpuCreateBindGroupLayout=wasmExports["emwgpuCreateBindGroupLayout"];_emwgpuCreateCommandBuffer=wasmExports["emwgpuCreateCommandBuffer"];_emwgpuCreateCommandEncoder=wasmExports["emwgpuCreateCommandEncoder"];_emwgpuCreateComputePassEncoder=wasmExports["emwgpuCreateComputePassEncoder"];_emwgpuCreateComputePipeline=wasmExports["emwgpuCreateComputePipeline"];_emwgpuCreateExternalTexture=wasmExports["emwgpuCreateExternalTexture"];_emwgpuCreatePipelineLayout=wasmExports["emwgpuCreatePipelineLayout"];_emwgpuCreateQuerySet=wasmExports["emwgpuCreateQuerySet"];_emwgpuCreateRenderBundle=wasmExports["emwgpuCreateRenderBundle"];_emwgpuCreateRenderBundleEncoder=wasmExports["emwgpuCreateRenderBundleEncoder"];_emwgpuCreateRenderPassEncoder=wasmExports["emwgpuCreateRenderPassEncoder"];_emwgpuCreateRenderPipeline=wasmExports["emwgpuCreateRenderPipeline"];_emwgpuCreateSampler=wasmExports["emwgpuCreateSampler"];_emwgpuCreateSurface=wasmExports["emwgpuCreateSurface"];_emwgpuCreateTexture=wasmExports["emwgpuCreateTexture"];_emwgpuCreateTextureView=wasmExports["emwgpuCreateTextureView"];_emwgpuCreateAdapter=wasmExports["emwgpuCreateAdapter"];_emwgpuCreateBuffer=wasmExports["emwgpuCreateBuffer"];_emwgpuCreateDevice=wasmExports["emwgpuCreateDevice"];_emwgpuCreateQueue=wasmExports["emwgpuCreateQueue"];_emwgpuCreateShaderModule=wasmExports["emwgpuCreateShaderModule"];_emwgpuOnDeviceLostCompleted=wasmExports["emwgpuOnDeviceLostCompleted"];_emwgpuOnMapAsyncCompleted=wasmExports["emwgpuOnMapAsyncCompleted"];_emwgpuOnRequestAdapterCompleted=wasmExports["emwgpuOnRequestAdapterCompleted"];_emwgpuOnRequestDeviceCompleted=wasmExports["emwgpuOnRequestDeviceCompleted"];_emwgpuOnWorkDoneCompleted=wasmExports["emwgpuOnWorkDoneCompleted"];_emwgpuOnUncapturedError=wasmExports["emwgpuOnUncapturedError"];__emscripten_tls_init=wasmExports["_emscripten_tls_init"];_pthread_self=wasmExports["pthread_self"];_emscripten_builtin_memalign=wasmExports["emscripten_builtin_memalign"];__emscripten_thread_init=wasmExports["_emscripten_thread_init"];__emscripten_thread_crashed=wasmExports["_emscripten_thread_crashed"];__emscripten_run_js_on_main_thread=wasmExports["_emscripten_run_js_on_main_thread"];__emscripten_thread_free_data=wasmExports["_emscripten_thread_free_data"];__emscripten_thread_exit=wasmExports["_emscripten_thread_exit"];__emscripten_check_mailbox=wasmExports["_emscripten_check_mailbox"];_memalign=wasmExports["memalign"];___trap=wasmExports["__trap"];_emscripten_stack_set_limits=wasmExports["emscripten_stack_set_limits"];__emscripten_stack_restore=wasmExports["_emscripten_stack_restore"];__emscripten_stack_alloc=wasmExports["_emscripten_stack_alloc"];_emscripten_stack_get_current=wasmExports["emscripten_stack_get_current"];__indirect_function_table=wasmTable=wasmExports["__indirect_function_table"]}var wasmImports;function assignWasmImports(){wasmImports={__asyncjs__js_file_read,__pthread_create_js:___pthread_create_js,__syscall_fcntl64:___syscall_fcntl64,__syscall_getcwd:___syscall_getcwd,__syscall_getdents64:___syscall_getdents64,__syscall_ioctl:___syscall_ioctl,__syscall_openat:___syscall_openat,__syscall_stat64:___syscall_stat64,_abort_js:__abort_js,_emscripten_init_main_thread_js:__emscripten_init_main_thread_js,_emscripten_notify_mailbox_postmessage:__emscripten_notify_mailbox_postmessage,_emscripten_receive_on_main_thread_js:__emscripten_receive_on_main_thread_js,_emscripten_thread_cleanup:__emscripten_thread_cleanup,_emscripten_thread_mailbox_await:__emscripten_thread_mailbox_await,_emscripten_thread_set_strongref:__emscripten_thread_set_strongref,_localtime_js:__localtime_js,_mmap_js:__mmap_js,_munmap_js:__munmap_js,_tzset_js:__tzset_js,clock_time_get:_clock_time_get,emscripten_check_blocking_allowed:_emscripten_check_blocking_allowed,emscripten_date_now:_emscripten_date_now,emscripten_exit_with_live_runtime:_emscripten_exit_with_live_runtime,emscripten_get_callstack:_emscripten_get_callstack,emscripten_get_heap_max:_emscripten_get_heap_max,emscripten_get_now:_emscripten_get_now,emscripten_has_asyncify:_emscripten_has_asyncify,emscripten_num_logical_cores:_emscripten_num_logical_cores,emscripten_resize_heap:_emscripten_resize_heap,emwgpuAdapterRequestDevice:_emwgpuAdapterRequestDevice,emwgpuBufferDestroy:_emwgpuBufferDestroy,emwgpuBufferGetConstMappedRange:_emwgpuBufferGetConstMappedRange,emwgpuBufferMapAsync:_emwgpuBufferMapAsync,emwgpuBufferUnmap:_emwgpuBufferUnmap,emwgpuDelete:_emwgpuDelete,emwgpuDeviceCreateBuffer:_emwgpuDeviceCreateBuffer,emwgpuDeviceCreateShaderModule:_emwgpuDeviceCreateShaderModule,emwgpuDeviceDestroy:_emwgpuDeviceDestroy,emwgpuInstanceRequestAdapter:_emwgpuInstanceRequestAdapter,emwgpuQueueOnSubmittedWorkDone:_emwgpuQueueOnSubmittedWorkDone,emwgpuWaitAny:_emwgpuWaitAny,environ_get:_environ_get,environ_sizes_get:_environ_sizes_get,exit:_exit,fd_close:_fd_close,fd_read:_fd_read,fd_seek:_fd_seek,fd_write:_fd_write,memory:wasmMemory,random_get:_random_get,wgpuAdapterGetInfo:_wgpuAdapterGetInfo,wgpuAdapterGetLimits:_wgpuAdapterGetLimits,wgpuAdapterHasFeature:_wgpuAdapterHasFeature,wgpuBufferGetSize:_wgpuBufferGetSize,wgpuCommandEncoderBeginComputePass:_wgpuCommandEncoderBeginComputePass,wgpuCommandEncoderCopyBufferToBuffer:_wgpuCommandEncoderCopyBufferToBuffer,wgpuCommandEncoderFinish:_wgpuCommandEncoderFinish,wgpuComputePassEncoderDispatchWorkgroups:_wgpuComputePassEncoderDispatchWorkgroups,wgpuComputePassEncoderEnd:_wgpuComputePassEncoderEnd,wgpuComputePassEncoderSetBindGroup:_wgpuComputePassEncoderSetBindGroup,wgpuComputePassEncoderSetPipeline:_wgpuComputePassEncoderSetPipeline,wgpuComputePipelineGetBindGroupLayout:_wgpuComputePipelineGetBindGroupLayout,wgpuDeviceCreateBindGroup:_wgpuDeviceCreateBindGroup,wgpuDeviceCreateCommandEncoder:_wgpuDeviceCreateCommandEncoder,wgpuDeviceCreateComputePipeline:_wgpuDeviceCreateComputePipeline,wgpuInstanceHasWGSLLanguageFeature:_wgpuInstanceHasWGSLLanguageFeature,wgpuQueueSubmit:_wgpuQueueSubmit,wgpuQueueWriteBuffer:_wgpuQueueWriteBuffer}}function applySignatureConversions(wasmExports){wasmExports=Object.assign({},wasmExports);var makeWrapper___PP=f=>(a0,a1,a2)=>f(a0,BigInt(a1?a1:0),BigInt(a2?a2:0));var makeWrapper_pp=f=>a0=>Number(f(BigInt(a0)));var makeWrapper__p=f=>a0=>f(BigInt(a0));var makeWrapper_p=f=>()=>Number(f());var makeWrapper_ppp=f=>(a0,a1)=>Number(f(BigInt(a0),BigInt(a1)));var makeWrapper__p_____=f=>(a0,a1,a2,a3,a4,a5)=>f(BigInt(a0),a1,a2,a3,a4,a5);var makeWrapper___p_p_=f=>(a0,a1,a2,a3,a4)=>f(a0,BigInt(a1),a2,BigInt(a3),a4);var makeWrapper__pp=f=>(a0,a1)=>f(BigInt(a0),BigInt(a1));wasmExports["main"]=makeWrapper___PP(wasmExports["main"]);wasmExports["malloc"]=makeWrapper_pp(wasmExports["malloc"]);wasmExports["free"]=makeWrapper__p(wasmExports["free"]);wasmExports["pthread_self"]=makeWrapper_p(wasmExports["pthread_self"]);wasmExports["emscripten_builtin_memalign"]=makeWrapper_ppp(wasmExports["emscripten_builtin_memalign"]);wasmExports["_emscripten_thread_init"]=makeWrapper__p_____(wasmExports["_emscripten_thread_init"]);wasmExports["_emscripten_run_js_on_main_thread"]=makeWrapper___p_p_(wasmExports["_emscripten_run_js_on_main_thread"]);wasmExports["_emscripten_thread_free_data"]=makeWrapper__p(wasmExports["_emscripten_thread_free_data"]);wasmExports["_emscripten_thread_exit"]=makeWrapper__p(wasmExports["_emscripten_thread_exit"]);wasmExports["memalign"]=makeWrapper_ppp(wasmExports["memalign"]);wasmExports["emscripten_stack_set_limits"]=makeWrapper__pp(wasmExports["emscripten_stack_set_limits"]);wasmExports["_emscripten_stack_restore"]=makeWrapper__p(wasmExports["_emscripten_stack_restore"]);wasmExports["_emscripten_stack_alloc"]=makeWrapper_pp(wasmExports["_emscripten_stack_alloc"]);wasmExports["emscripten_stack_get_current"]=makeWrapper_p(wasmExports["emscripten_stack_get_current"]);return wasmExports}async function callMain(){var entryFunction=_main;var argc=0;var argv=0;try{var ret=entryFunction(argc,BigInt(argv));ret=await ret;exitJS(ret,true);return ret}catch(e){return handleException(e)}}function run(){if(runDependencies>0){dependenciesFulfilled=run;return}if(ENVIRONMENT_IS_PTHREAD){initRuntime();return}preRun();if(runDependencies>0){dependenciesFulfilled=run;return}async function doRun(){Module["calledRun"]=true;if(ABORT)return;initRuntime();preMain();Module["onRuntimeInitialized"]?.();var noInitialRun=Module["noInitialRun"]||false;if(!noInitialRun)await callMain();postRun()}if(Module["setStatus"]){Module["setStatus"]("Running...");setTimeout(()=>{setTimeout(()=>Module["setStatus"](""),1);doRun()},1)}else{doRun()}}var wasmExports;if(!ENVIRONMENT_IS_PTHREAD){createWasm();run()}\n';

// src/worker.ts
var FILE_READ_REQ_EVENT = "fs.read_req";
var JSPI_STUB = `
if (!WebAssembly.Suspending) {
  // JSPI not available - stubs that keep the import/export tables valid.
  // Suspending wraps imports: identity is fine since async imports won't be called.
  WebAssembly.Suspending = function (fn) {
    // console.log(fn.toString());
    return fn;
  };
  // promising wraps exports: must return a Promise so ccall's ret.then() works.
  WebAssembly.promising = function (fn) {
    return function (...args) {
      try {
        return Promise.resolve(fn(...args));
      } catch (e) {
        return Promise.reject(e);
      }
    };
  };
}
`;
var ProxyToWorker = class {
  // filename -> Blob for async reads
  constructor(resources, nbThread, suppressNativeLog, logger) {
    __publicField(this, "resources");
    __publicField(this, "logger");
    __publicField(this, "suppressNativeLog");
    __publicField(this, "taskQueue", []);
    __publicField(this, "taskId", 1);
    __publicField(this, "resultQueue", []);
    __publicField(this, "busy", false);
    // is the work loop is running?
    __publicField(this, "worker");
    __publicField(this, "multiThread");
    __publicField(this, "nbThread");
    __publicField(this, "useAsyncFile");
    __publicField(this, "fileBlobs", /* @__PURE__ */ new Map());
    this.resources = resources;
    this.nbThread = nbThread;
    this.multiThread = nbThread > 0;
    this.logger = logger;
    this.suppressNativeLog = suppressNativeLog;
    this.useAsyncFile = canUseAsyncFileRead(resources.compat);
  }
  getModuleCode() {
    return __async(this, null, function* () {
      if (!this.resources.jsPath) {
        if (this.resources.compat) {
          throw new Error(
            "compat mode is enabled but no jsPath was provided. Pass a worker JS via setCompat() or install @wllama/wllama-compat."
          );
        }
        return WLLAMA_EMSCRIPTEN_CODE;
      } else if (this.resources.jsPath.code) {
        return this.resources.jsPath.code;
      } else if (isString(this.resources.jsPath)) {
        const response = yield fetch(this.resources.jsPath);
        if (!response.ok) {
          throw new Error(
            `Failed to fetch worker code from ${this.resources.jsPath}`
          );
        }
        return yield response.text();
      } else {
        throw new Error("No JS code provided for worker");
      }
    });
  }
  moduleInit(ggufFiles) {
    return __async(this, null, function* () {
      let moduleCode = JSPI_STUB + (yield this.getModuleCode());
      if (this.resources.noWebGPU) {
        moduleCode = 'try{Object.defineProperty(WorkerNavigator.prototype,"gpu",{get:()=>({requestAdapter:async()=>null})});}catch(e){}' + moduleCode;
      }
      let mainModuleCode = moduleCode.replace("var Module", "var ___Module");
      const runOptions = {
        pathConfig: {
          "wllama.wasm": this.resources.wasmPath
        },
        nbThread: this.nbThread,
        compat: this.resources.compat
      };
      const completeCode = [
        `const RUN_OPTIONS = ${JSON.stringify(runOptions)};`,
        `function wModuleInit() { ${mainModuleCode}; return Module; }`,
        LLAMA_CPP_WORKER_CODE
      ].join(";\n\n");
      this.worker = createWorker(completeCode);
      this.worker.onmessage = this.onRecvMsg.bind(this);
      this.worker.onerror = this.logger.error;
      const res = yield this.pushTask({
        verb: "module.init",
        args: [
          new Blob([moduleCode], { type: "text/javascript" }),
          this.useAsyncFile
        ],
        callbackId: this.taskId++
      });
      const nativeFiles = [];
      for (const file of ggufFiles) {
        const needAllocBuffer = !this.useAsyncFile;
        const id = yield this.fileAlloc(
          file.name,
          file.blob.size,
          needAllocBuffer
        );
        nativeFiles.push(__spreadValues({ id }, file));
        if (this.useAsyncFile) {
          this.fileBlobs.set(file.name, file.blob);
        }
      }
      if (!this.useAsyncFile) {
        yield Promise.all(
          nativeFiles.map((file) => {
            return this.fileWrite(file.id, file.blob);
          })
        );
      }
      return res;
    });
  }
  wllamaStart() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.start",
        args: [],
        callbackId: this.taskId++
      });
      const parsedResult = this.parseResult(result);
      return parsedResult;
    });
  }
  wllamaAction(name, body) {
    return __async(this, null, function* () {
      const encodedMsg = glueSerialize(body);
      const result = yield this.pushTask({
        verb: "wllama.action",
        args: [name, encodedMsg],
        callbackId: this.taskId++
      });
      const parsedResult = glueDeserialize(result);
      if (parsedResult._name === "erro_evt") {
        throw new WllamaRuntimeError(parsedResult.message, "");
      }
      return parsedResult;
    });
  }
  /** Take a slot save file out of the worker's in-memory FS (the file is removed) */
  slotFileRead(fileName) {
    return __async(this, null, function* () {
      return yield this.pushTask({
        verb: "fs.slot_read",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  /** Put a slot save file into the worker's in-memory FS */
  slotFileWrite(fileName, data) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_write",
        args: [fileName, data],
        callbackId: this.taskId++
      });
    });
  }
  slotFileRemove(fileName) {
    return __async(this, null, function* () {
      yield this.pushTask({
        verb: "fs.slot_remove",
        args: [fileName],
        callbackId: this.taskId++
      });
    });
  }
  wllamaExit() {
    return __async(this, null, function* () {
      if (this.worker) {
        this.worker.terminate();
      }
    });
  }
  wllamaDebug() {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "wllama.debug",
        args: [],
        callbackId: this.taskId++
      });
      return JSON.parse(result);
    });
  }
  ///////////////////////////////////////
  /**
   * Allocate a new file in heapfs
   * @returns fileId, to be used by fileWrite()
   */
  fileAlloc(fileName, size, allocBuffer) {
    return __async(this, null, function* () {
      const result = yield this.pushTask({
        verb: "fs.alloc",
        args: [fileName, size, allocBuffer],
        callbackId: this.taskId++
      });
      return result.fileId;
    });
  }
  /**
   * Write a Blob to heapfs
   */
  fileWrite(fileId, blob) {
    return __async(this, null, function* () {
      const reader = blob.stream().getReader();
      let offset = 0;
      while (true) {
        const { done, value } = yield reader.read();
        if (done) break;
        const size = value.byteLength;
        yield this.pushTask(
          {
            verb: "fs.write",
            args: [fileId, value, offset],
            callbackId: this.taskId++
          },
          // @ts-ignore Type 'ArrayBufferLike' is not assignable to type 'ArrayBuffer'
          [value.buffer]
        );
        offset += size;
      }
    });
  }
  fileReadResponse(name, offset, size) {
    return __async(this, null, function* () {
      var _a;
      try {
        const blob = this.fileBlobs.get(name);
        if (!blob) {
          throw new Error(`blob not found for name="${name}"`);
        }
        const chunk = blob.slice(offset, offset + size);
        const buffer = yield chunk.arrayBuffer();
        this.worker.postMessage(
          { verb: "fs.read_res", args: [buffer] },
          { transfer: [buffer] }
        );
      } catch (err) {
        this.logger.error("fileReadResponse failed, terminating worker:", err);
        (_a = this.worker) == null ? void 0 : _a.terminate();
        this.worker = void 0;
        this.abort(`File read failed: ${err}`, err.stack || "");
      }
    });
  }
  /**
   * Parse JSON result returned by cpp code.
   * Throw new Error if "__exception" is present in the response
   *
   * TODO: get rid of this function once everything is migrated to Glue
   */
  parseResult(result) {
    const parsedResult = JSON.parse(result);
    if (parsedResult && parsedResult["error"]) {
      throw new WllamaRuntimeError("Unknown error, please see console.log", "");
    }
    return parsedResult;
  }
  /**
   * Push a new task to taskQueue
   */
  pushTask(param, buffers) {
    return new Promise((resolve, reject) => {
      this.taskQueue.push({ resolve, reject, param, buffers });
      this.runTaskLoop();
    });
  }
  /**
   * Main loop for processing tasks
   */
  runTaskLoop() {
    return __async(this, null, function* () {
      var _a;
      if (this.busy) {
        return;
      }
      this.busy = true;
      while (true) {
        const task = this.taskQueue.shift();
        if (!task) break;
        this.resultQueue.push(task);
        this.worker.postMessage(
          task.param,
          isSafariMobile() ? void 0 : {
            transfer: (_a = task.buffers) != null ? _a : []
          }
        );
      }
      this.busy = false;
    });
  }
  /**
   * Handle messages from worker
   */
  onRecvMsg(e) {
    if (!e.data) return;
    const { verb, args } = e.data;
    const isCompatBuild = this.resources.compat;
    if (verb && verb.startsWith("console.")) {
      if (this.suppressNativeLog) {
        return;
      }
      if (verb.endsWith("debug")) this.logger.debug(...args);
      if (verb.endsWith("log")) this.logger.log(...args);
      if (verb.endsWith("warn")) this.logger.warn(...args);
      if (verb.endsWith("error")) this.logger.error(...args);
      return;
    } else if (verb === "signal.abort") {
      const [signalType, message, rawStack, originalErr] = args;
      if (originalErr) {
        this.logger.error(originalErr);
      }
      (() => __async(this, null, function* () {
        let stack = "";
        let newMsg = message.replace(
          "Build with -sASSERTIONS for more info.",
          ""
        );
        if (signalType === "abort") {
          newMsg = `(ABORT) ${newMsg}`;
          stack = rawStack.replace(/\|/g, "\n");
        } else if (signalType === "exception") {
          stack = rawStack;
        }
        const decoded = yield Debug.decodeStackTrace(stack, isCompatBuild);
        this.logger.error(`Stack trace (${signalType}):
` + decoded);
        this.abort(newMsg, decoded);
      }))();
      return;
    }
    if (verb === FILE_READ_REQ_EVENT) {
      const [name, offset, size] = args;
      this.fileReadResponse(name, offset, size).catch(() => {
      });
      return;
    }
    const { callbackId, result, err } = e.data;
    if (callbackId) {
      const idx = this.resultQueue.findIndex(
        (t) => t.param.callbackId === callbackId
      );
      if (idx !== -1) {
        const waitingTask = this.resultQueue.splice(idx, 1)[0];
        if (err) waitingTask.reject(err);
        else waitingTask.resolve(result);
      } else {
        this.logger.error(
          `Cannot find waiting task with callbackId = ${callbackId}`
        );
      }
    }
  }
  abort(text, stack) {
    const error = new WllamaRuntimeError(
      text.length == 0 ? "(unknown error)" : text,
      stack
    );
    while (this.resultQueue.length > 0) {
      const waitingTask = this.resultQueue.pop();
      if (!waitingTask) break;
      waitingTask.reject(error);
    }
    while (this.taskQueue.length > 0) {
      const pendingTask = this.taskQueue.pop();
      if (!pendingTask) break;
      pendingTask.reject(error);
    }
  }
};

// src/huggingface.ts
var HF_BASE = "https://huggingface.co";
var DEFAULT_QUANTS = ["Q4_K_M", "Q8_0"];
function fetchRepoFiles(repo, token) {
  return __async(this, null, function* () {
    var _a;
    const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
    const headers = { Accept: "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    const res = yield fetch(url, { headers });
    if (!res.ok) {
      let msg = res.statusText;
      try {
        msg = (_a = (yield res.json()).error) != null ? _a : msg;
      } catch (e) {
      }
      throw new Error(`HF API error (${res.status}): ${msg}`);
    }
    return res.json();
  });
}
function firstShardPath(files, path) {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}
function selectFile(files, quant, mmprojOnly) {
  const candidates = files.filter((f) => {
    if (f.type !== "file" || !f.path.toLowerCase().endsWith(".gguf"))
      return false;
    const ismmproj = f.path.toLowerCase().includes("mmproj");
    return mmprojOnly ? ismmproj : !ismmproj;
  });
  if (candidates.length === 0) return null;
  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }
  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }
  return firstShardPath(candidates, candidates[0].path);
}
function getHFModelSource(config) {
  return __async(this, null, function* () {
    const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;
    const files = yield fetchRepoFiles(repo, hfToken);
    const modelPath = file != null ? file : selectFile(files, quant, false);
    if (!modelPath) {
      throw new Error(`No GGUF file found in repo "${repo}"`);
    }
    const source = {
      url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`
    };
    if (mmprojFile || mmprojQuant !== void 0) {
      const mmpath = mmprojFile != null ? mmprojFile : selectFile(files, mmprojQuant, true);
      if (mmpath) {
        source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
      }
    }
    if (hfToken) {
      const params = new URLSearchParams({ token: hfToken });
      source.url += `?${params}`;
      if (source.mmprojUrl) {
        source.mmprojUrl += `?${params}`;
      }
    }
    return source;
  });
}
function getHFFileSHA256(url, headers) {
  return __async(this, null, function* () {
    if (!url.includes("/resolve/")) return void 0;
    const rawUrl = url.replace("/resolve/", "/raw/");
    try {
      const text = yield fetch(rawUrl, { headers }).then((r) => r.text());
      const match = text.match(/^oid sha256:([0-9a-f]{64})$/m);
      return match ? match[1] : void 0;
    } catch (e) {
      return void 0;
    }
  });
}

// src/storage/opfs.ts
var OPFSBackend = class {
  isSupported() {
    var _a;
    return typeof navigator !== "undefined" && "storage" in navigator && !!((_a = navigator.storage) == null ? void 0 : _a.getDirectory);
  }
  read(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        return yield fileHandle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  write(key, stream) {
    return __async(this, null, function* () {
      const writable = yield openWritable(key);
      yield writable.truncate(0);
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        const fileHandle = yield cacheDir.getFileHandle(key);
        const file = yield fileHandle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      const cacheDir = yield getCacheDir();
      const result = [];
      try {
        for (var iter = __forAwait(cacheDir.entries()), more, temp, error; more = !(temp = yield iter.next()).done; more = false) {
          const [name, handle] = temp.value;
          if (handle.kind === "file") {
            const file = yield handle.getFile();
            result.push({ key: name, size: file.size });
          }
        }
      } catch (temp) {
        error = [temp];
      } finally {
        try {
          more && (temp = iter.return) && (yield temp.call(iter));
        } finally {
          if (error)
            throw error[0];
        }
      }
      return result;
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      try {
        const cacheDir = yield getCacheDir();
        yield cacheDir.removeEntry(key);
      } catch (e) {
        if ((e == null ? void 0 : e.name) !== "NotFoundError") throw e;
      }
    });
  }
};
function getCacheDir() {
  return __async(this, null, function* () {
    const opfsRoot = yield navigator.storage.getDirectory();
    return opfsRoot.getDirectoryHandle("cache", { create: true });
  });
}
function openWritable(fileName) {
  return __async(this, null, function* () {
    const worker = createWorker(OPFS_UTILS_WORKER_CODE);
    let pResolve;
    let pReject;
    worker.onmessage = (e) => {
      if (e.data.ok) pResolve(null);
      else if (e.data.err) pReject(e.data.err);
    };
    worker.onerror = (e) => {
      var _a;
      return pReject == null ? void 0 : pReject((_a = e.message) != null ? _a : e);
    };
    const workerExec = (data) => new Promise((resolve, reject) => {
      pResolve = resolve;
      pReject = reject;
      worker.postMessage(
        data,
        isSafariMobile() ? void 0 : { transfer: "buf" in data && data.buf ? [data.buf.buffer] : [] }
      );
    });
    yield workerExec({ action: "open", filename: fileName });
    return {
      truncate: () => __async(this, null, function* () {
      }),
      write: (value) => workerExec({ action: "write", buf: value }),
      close: () => __async(this, null, function* () {
        yield workerExec({ action: "close" });
        worker.terminate();
      })
    };
  });
}

// src/storage/cos.ts
function makeHash(key) {
  return { algorithm: "SHA-256", value: key };
}
var COSInternalBackend = class {
  isSupported() {
    return typeof navigator !== "undefined" && "crossOriginStorage" in navigator;
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  read(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        return handle.getFile();
      } catch (e) {
        return null;
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  write(key, stream) {
    return __async(this, null, function* () {
      const handle = yield navigator.crossOriginStorage.requestFileHandle(
        makeHash(key),
        { create: true }
      );
      const writable = yield handle.createWritable();
      const reader = stream.getReader();
      try {
        while (true) {
          const { done, value } = yield reader.read();
          if (done) break;
          yield writable.write(value);
        }
      } finally {
        yield writable.close();
      }
    });
  }
  // IMPORTANT: key must be SHA-256 hash of the data
  getSize(key) {
    return __async(this, null, function* () {
      try {
        const handle = yield navigator.crossOriginStorage.requestFileHandle(
          makeHash(key)
        );
        const file = yield handle.getFile();
        return file.size;
      } catch (e) {
        return -1;
      }
    });
  }
  list() {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
  delete(_key) {
    return __async(this, null, function* () {
      throw new Error("not implemented");
    });
  }
};
var COSBackend = class {
  constructor() {
    __publicField(this, "cos", new COSInternalBackend());
    __publicField(this, "priv", new OPFSBackend());
  }
  isSupported() {
    return this.priv.isSupported();
  }
  read(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const blob = yield this.cos.read(hint.sha256);
        if (blob) return blob;
      }
      return this.priv.read(key);
    });
  }
  write(key, stream, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        yield this.cos.write(hint.sha256, stream);
      } else {
        yield this.priv.write(key, stream);
      }
    });
  }
  getSize(key, hint) {
    return __async(this, null, function* () {
      if ((hint == null ? void 0 : hint.sha256) && this.cos.isSupported()) {
        const size = yield this.cos.getSize(hint.sha256);
        if (size !== -1) return size;
      }
      return this.priv.getSize(key);
    });
  }
  list() {
    return __async(this, null, function* () {
      return this.priv.list();
    });
  }
  delete(key) {
    return __async(this, null, function* () {
      return this.priv.delete(key);
    });
  }
};

// src/cache-manager.ts
var PREFIX_METADATA = "__metadata__";
var POLYFILL_ETAG = "polyfill_for_older_version";
function hintFromMetadata(metadata) {
  if (!metadata) return void 0;
  if (metadata.sha256) return { sha256: metadata.sha256 };
  return void 0;
}
var CacheManager = class {
  /**
   * @param backends Array of storage backends to use, in order of preference ; if first is available, use it, otherwise try the next one.
   */
  constructor(backends = [new COSBackend()]) {
    __publicField(this, "sb");
    for (const backend of backends) {
      if (backend.isSupported()) {
        this.sb = backend;
        return;
      }
    }
    throw new Error("No supported storage backend found");
  }
  /**
   * Convert a given URL into a storage key.
   *
   * Format: `${hashSHA1(fullURL)}_${fileName}`
   */
  getNameFromURL(url) {
    return __async(this, null, function* () {
      return urlToFileName(url, "");
    });
  }
  /**
   * @deprecated Use `download()` instead
   *
   * Write a new file to cache. This will overwrite existing file.
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   */
  write(name, stream, metadata) {
    return __async(this, null, function* () {
      yield this.sb.write(name, stream);
      yield this.writeMetadata(name, metadata);
    });
  }
  download(_0) {
    return __async(this, arguments, function* (url, options = {}) {
      var _a, _b, _c, _d;
      const fileKey = yield urlToFileName(url, "");
      const sha256 = yield getHFFileSHA256(url, (_a = options.headers) != null ? _a : {});
      const hint = sha256 ? { sha256 } : void 0;
      const cachedSize = yield this.sb.getSize(fileKey, hint);
      if (cachedSize !== -1) {
        const metadata2 = yield this.readMetadata(fileKey);
        if ((metadata2 == null ? void 0 : metadata2.originalURL) === url && metadata2.originalSize === cachedSize) {
          return;
        }
        const head = yield fetch(url, __spreadValues({
          method: "HEAD"
        }, options.headers ? { headers: options.headers } : {}));
        const originalSize = parseInt(
          (_b = head.headers.get("content-length")) != null ? _b : "0",
          10
        );
        const etag2 = (head.headers.get("etag") || "").replace(
          /[^A-Za-z0-9]/g,
          ""
        );
        if (originalSize > 0 && originalSize === cachedSize) {
          yield this.writeMetadata(fileKey, __spreadValues({
            originalURL: url,
            originalSize,
            etag: etag2,
            sha256
          }, (_c = options.metadataAdditional) != null ? _c : {}));
          return;
        }
        yield this.sb.delete(fileKey);
        yield this.sb.delete(`${PREFIX_METADATA}${fileKey}`);
      }
      const response = yield fetch(url, __spreadValues(__spreadValues({}, options.headers ? { headers: options.headers } : {}), options.signal ? { signal: options.signal } : {}));
      if (!response.ok || !response.body) {
        throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
      }
      const contentLength = response.headers.get("content-length");
      const etag = (response.headers.get("etag") || "").replace(
        /[^A-Za-z0-9]/g,
        ""
      );
      const total = parseInt(contentLength != null ? contentLength : "0", 10);
      const progressCallback = options.progressCallback;
      let loaded = 0;
      let lastProgressAt = 0;
      const progressStream = new TransformStream({
        transform(chunk, controller) {
          loaded += chunk.byteLength;
          if (progressCallback) {
            const now = Date.now();
            if (now - lastProgressAt > 100) {
              lastProgressAt = now;
              progressCallback({ loaded, total });
            }
          }
          controller.enqueue(chunk);
        },
        flush() {
          progressCallback == null ? void 0 : progressCallback({ loaded, total: total || loaded });
        }
      });
      const metadata = __spreadValues({
        originalURL: url,
        originalSize: total,
        etag
      }, (_d = options.metadataAdditional) != null ? _d : {});
      if (sha256) {
        metadata.sha256 = sha256;
      }
      yield this.sb.write(
        fileKey,
        response.body.pipeThrough(progressStream),
        hint
      );
      yield this.writeMetadata(fileKey, metadata);
    });
  }
  /**
   * Open a file in cache for reading
   *
   * @param nameOrURL The file name returned by `getNameFromURL()` or `list()`, or the original URL of the remote file
   * @returns Blob, or null if file does not exist
   */
  open(nameOrURL) {
    return __async(this, null, function* () {
      const hint1 = hintFromMetadata(yield this.getMetadata(nameOrURL));
      const direct = yield this.sb.read(nameOrURL, hint1);
      if (direct) return direct;
      const key = yield urlToFileName(nameOrURL, "");
      const hint2 = hintFromMetadata(yield this.getMetadata(key));
      return this.sb.read(key, hint2);
    });
  }
  /**
   * Get the size of a file in stored cache
   *
   * NOTE: in case the download is stopped mid-way (i.e. user close browser tab), the file maybe corrupted, size maybe different from `metadata.originalSize`
   *
   * @param name The file name returned by `getNameFromURL()` or `list()`
   * @returns number of bytes, or -1 if file does not exist
   */
  getSize(name) {
    return __async(this, null, function* () {
      const hint = hintFromMetadata(yield this.getMetadata(name));
      return this.sb.getSize(name, hint);
    });
  }
  /**
   * Get metadata of a cached file
   */
  getMetadata(name) {
    return __async(this, null, function* () {
      const metadata = yield this.readMetadata(name);
      if (metadata) return metadata;
      const cachedSize = yield this.sb.getSize(name);
      return cachedSize > 0 ? (
        // files created by older version of wllama don't have metadata; polyfill it
        {
          etag: POLYFILL_ETAG,
          originalSize: cachedSize,
          originalURL: ""
        }
      ) : (
        // cached file not found
        null
      );
    });
  }
  /**
   * Same as `getMetadata()`, but without polyfill. Returns null if the file has no metadata.
   */
  readMetadata(name) {
    return __async(this, null, function* () {
      const blob = yield this.sb.read(`${PREFIX_METADATA}${name}`);
      if (!blob) return null;
      try {
        return yield new Response(blob).json();
      } catch (e) {
        return null;
      }
    });
  }
  /**
   * List all files currently in cache
   */
  list() {
    return __async(this, null, function* () {
      const all = yield this.sb.list();
      const metadataMap = {};
      for (const { key } of all) {
        if (key.startsWith(PREFIX_METADATA)) {
          const blob = yield this.sb.read(key);
          if (blob) {
            const meta = yield new Response(blob).json().catch(() => null);
            metadataMap[key.slice(PREFIX_METADATA.length)] = meta;
          }
        }
      }
      const result = [];
      for (const { key, size } of all) {
        if (!key.startsWith(PREFIX_METADATA)) {
          result.push({
            name: key,
            size,
            metadata: metadataMap[key] || {
              originalSize: size,
              originalURL: "",
              etag: ""
            }
          });
        }
      }
      return result;
    });
  }
  /**
   * Clear all files currently in cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.deleteMany(() => true);
    });
  }
  /**
   * Delete a single file in cache
   *
   * @param nameOrURL Can be either an URL or a name returned by `getNameFromURL()` or `list()`
   */
  delete(nameOrURL) {
    return __async(this, null, function* () {
      const name2 = yield this.getNameFromURL(nameOrURL);
      yield this.deleteMany(
        (entry) => entry.name === nameOrURL || entry.name === name2
      );
    });
  }
  /**
   * Delete multiple files in cache.
   *
   * @param predicate A predicate like `array.filter(item => boolean)`
   */
  deleteMany(predicate) {
    return __async(this, null, function* () {
      const list = yield this.list();
      for (const item of list) {
        if (predicate(item)) {
          yield this.sb.delete(item.name);
          yield this.sb.delete(`${PREFIX_METADATA}${item.name}`);
        }
      }
    });
  }
  /**
   * Write the metadata of the file to disk.
   */
  writeMetadata(name, metadata) {
    return __async(this, null, function* () {
      const blob = new Blob([JSON.stringify(metadata)], { type: "text/plain" });
      yield this.sb.write(`${PREFIX_METADATA}${name}`, blob.stream());
    });
  }
};
var cache_manager_default = CacheManager;
function urlToFileName(url, prefix) {
  return __async(this, null, function* () {
    const hashBuffer = yield crypto.subtle.digest(
      "SHA-1",
      new TextEncoder().encode(url)
    );
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
    return `${prefix}${hashHex}_${url.split("/").pop()}`;
  });
}

// src/model-manager.ts
var DEFAULT_PARALLEL_DOWNLOADS = 3;
var ModelValidationStatus = /* @__PURE__ */ ((ModelValidationStatus2) => {
  ModelValidationStatus2["VALID"] = "valid";
  ModelValidationStatus2["INVALID"] = "invalid";
  ModelValidationStatus2["DELETED"] = "deleted";
  return ModelValidationStatus2;
})(ModelValidationStatus || {});
var Model = class {
  constructor(modelManager, url, mmprojUrl, savedFiles) {
    __publicField(this, "modelManager");
    /**
     * URL to the GGUF file (in case it contains multiple shards, the URL should point to the first shard)
     *
     * This URL will be used to identify the model in the cache. There can't be 2 models with the same URL.
     */
    __publicField(this, "url");
    /**
     * URL to mmproj file, if exists
     */
    __publicField(this, "mmprojUrl");
    /**
     * Size in bytes (total size of all shards).
     *
     * A value of -1 means the model is deleted from the cache. You must call `ModelManager.downloadModel` to re-download the model.
     */
    __publicField(this, "size");
    /**
     * List of all shards in the cache, sorted by original URL (ascending order)
     */
    __publicField(this, "files");
    this.modelManager = modelManager;
    this.url = url;
    this.mmprojUrl = mmprojUrl;
    if (savedFiles) {
      this.files = this.getAllFiles(savedFiles);
      this.size = sumArr(this.files.map((f) => f.metadata.originalSize));
    } else {
      this.files = [];
      this.size = 0;
    }
  }
  /**
   * Open and get a list of all shards as Blobs
   */
  open() {
    return __async(this, null, function* () {
      if (this.size === -1) {
        throw new WllamaError(
          `Model is deleted from the cache; Call ModelManager.downloadModel to re-download the model`,
          "load_error"
        );
      }
      const blobs = [];
      for (const file of this.files) {
        const blob = yield this.modelManager.cacheManager.open(file.name);
        if (!blob) {
          throw new Error(
            `Failed to open file ${file.name}; Hint: the model may be invalid, please refresh it`
          );
        }
        blobs.push(blob);
      }
      return blobs;
    });
  }
  /**
   * Validate the model files.
   *
   * If the model is invalid, the model manager will not be able to use it. You must call `refresh` to re-download the model.
   *
   * Cases that model is invalid:
   * - The model is deleted from the cache
   * - The model files are missing (or the download is interrupted)
   */
  validate() {
    let nbShards = ModelManager.parseModelUrl(this.url).length;
    if (this.mmprojUrl) {
      nbShards += 1;
    }
    if (this.size === -1) {
      return "deleted" /* DELETED */;
    }
    if (this.size < 16 || this.files.length !== nbShards) {
      return "invalid" /* INVALID */;
    }
    for (const file of this.files) {
      if (!file.metadata || file.metadata.originalSize !== file.size) {
        return "invalid" /* INVALID */;
      }
    }
    return "valid" /* VALID */;
  }
  /**
   * In case the model is invalid, call this function to re-download the model
   */
  refresh() {
    return __async(this, arguments, function* (options = {}) {
      var _a;
      const urls = ModelManager.parseModelUrl(this.url);
      if (this.mmprojUrl) {
        urls.push(this.mmprojUrl);
      }
      const works = urls.map((url, index) => ({
        url,
        index
      }));
      this.modelManager.logger.debug("Downloading model files:", urls);
      const nParallel = (_a = this.modelManager.params.parallelDownloads) != null ? _a : DEFAULT_PARALLEL_DOWNLOADS;
      const totalSize = yield this.getTotalDownloadSize(urls);
      const loadedSize = [];
      const worker = () => __async(this, null, function* () {
        while (works.length > 0) {
          const w = works.shift();
          if (!w) break;
          yield this.modelManager.cacheManager.download(w.url, __spreadProps(__spreadValues({}, options), {
            metadataAdditional: {
              originalURL: w.url,
              mmprojURL: this.mmprojUrl
            },
            progressCallback: ({ loaded }) => {
              var _a2;
              loadedSize[w.index] = loaded;
              (_a2 = options.progressCallback) == null ? void 0 : _a2.call(options, {
                loaded: sumArr(loadedSize),
                total: totalSize
              });
            }
          }));
        }
      });
      const promises = [];
      for (let i = 0; i < nParallel; i++) {
        promises.push(worker());
        loadedSize.push(0);
      }
      yield Promise.all(promises);
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      this.size = this.files.reduce((acc, f) => acc + f.metadata.originalSize, 0);
    });
  }
  /**
   * Remove the model from the cache
   */
  remove() {
    return __async(this, null, function* () {
      this.files = this.getAllFiles(yield this.modelManager.cacheManager.list());
      yield this.modelManager.cacheManager.deleteMany(
        (f) => !!this.files.find((file) => file.name === f.name)
      );
      this.size = -1;
    });
  }
  getAllFiles(savedFiles) {
    const allUrls = new Set(ModelManager.parseModelUrl(this.url));
    if (this.mmprojUrl) {
      allUrls.add(this.mmprojUrl);
    }
    const allFiles = [];
    for (const url of allUrls) {
      const file = savedFiles.find((f) => f.metadata.originalURL === url);
      if (!file) {
        throw new Error(`Model file not found: ${url}`);
      }
      allFiles.push(file);
    }
    allFiles.sort(
      (a, b) => a.metadata.originalURL.localeCompare(b.metadata.originalURL)
    );
    return allFiles;
  }
  getTotalDownloadSize(urls) {
    return __async(this, null, function* () {
      const responses = yield Promise.all(
        urls.map((url) => fetch(url, { method: "HEAD" }))
      );
      const sizes = responses.map(
        (res) => Number(res.headers.get("content-length") || "0")
      );
      return sumArr(sizes);
    });
  }
};
var ModelManager = class _ModelManager {
  constructor(params = {}) {
    // The CacheManager singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "params");
    __publicField(this, "logger");
    this.cacheManager = params.cacheManager || new cache_manager_default();
    this.params = params;
    this.logger = params.logger || console;
  }
  /**
   * Parses a model URL and returns an array of URLs based on the following patterns:
   * - If the input URL is an array, it returns the array itself.
   * - If the input URL is a string in the `gguf-split` format, it returns an array containing the URL of each shard in ascending order.
   * - Otherwise, it returns an array containing the input URL as a single element array.
   * @param modelUrl URL or list of URLs
   */
  static parseModelUrl(modelUrl) {
    var _a;
    if (Array.isArray(modelUrl)) {
      return modelUrl;
    }
    const urlPartsRegex = /-(\d{5})-of-(\d{5})\.gguf(?:\?.*)?$/;
    const queryMatch = modelUrl.match(/\.gguf(\?.*)?$/);
    const queryParams = (_a = queryMatch == null ? void 0 : queryMatch[1]) != null ? _a : "";
    const matches = modelUrl.match(urlPartsRegex);
    if (!matches) {
      return [modelUrl];
    }
    const baseURL = modelUrl.replace(urlPartsRegex, "");
    const total = matches[2];
    const paddedShardIds = Array.from(
      { length: Number(total) },
      (_, index) => (index + 1).toString().padStart(5, "0")
    );
    return paddedShardIds.map(
      (current) => `${baseURL}-${current}-of-${total}.gguf${queryParams}`
    );
  }
  /**
   * Get all models in the cache
   */
  getModels() {
    return __async(this, arguments, function* (opts = {}) {
      const cachedFiles = yield this.cacheManager.list();
      let models = [];
      for (const file of cachedFiles) {
        if (!file.metadata.originalURL) continue;
        const shards = _ModelManager.parseModelUrl(file.metadata.originalURL);
        const mmprojUrl = file.metadata.mmprojURL;
        const isFirstShard = shards.length === 1 || shards[0] === file.metadata.originalURL;
        if (isFirstShard) {
          models.push(
            new Model(this, file.metadata.originalURL, mmprojUrl, cachedFiles)
          );
        }
      }
      if (!opts.includeInvalid) {
        models = models.filter(
          (m) => m.validate() === "valid" /* VALID */
        );
      }
      return models;
    });
  }
  /**
   * Download a model from the given URL.
   *
   * The URL must end with `.gguf`
   */
  downloadModel(_0) {
    return __async(this, arguments, function* (sourceOrURL, options = {}) {
      const source = isString(sourceOrURL) ? { url: sourceOrURL } : sourceOrURL;
      if (!isValidGgufFile(source.url)) {
        throw new WllamaError(
          `Invalid model URL: ${source.url}; URL must ends with ".gguf"`,
          "download_error"
        );
      }
      const model = new Model(this, source.url, source.mmprojUrl);
      const validity = model.validate();
      if (validity !== "valid" /* VALID */) {
        yield model.refresh(options);
      }
      return model;
    });
  }
  /**
   * Get a model from the cache or download it if it's not available.
   */
  getModelOrDownload(_0) {
    return __async(this, arguments, function* (source, options = {}) {
      var _a;
      const models = yield this.getModels();
      const model = models.find((m) => m.url === source.url);
      if (model) {
        (_a = options.progressCallback) == null ? void 0 : _a.call(options, { loaded: model.size, total: model.size });
        return model;
      }
      return this.downloadModel(source, options);
    });
  }
  /**
   * Remove all models from the cache
   */
  clear() {
    return __async(this, null, function* () {
      yield this.cacheManager.clear();
    });
  }
};

// src/types/types.ts
var LogLevel = /* @__PURE__ */ ((LogLevel2) => {
  LogLevel2[LogLevel2["DEBUG"] = 1] = "DEBUG";
  LogLevel2[LogLevel2["INFO"] = 2] = "INFO";
  LogLevel2[LogLevel2["WARN"] = 3] = "WARN";
  LogLevel2[LogLevel2["ERROR"] = 4] = "ERROR";
  return LogLevel2;
})(LogLevel || {});

// src/wasm-from-cdn.ts
var WasmCompatFromCDN = {
  worker: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.js",
  wasm: "https://cdn.jsdelivr.net/npm/@wllama/wllama-compat@3.12.0/wasm/wllama.wasm"
};

// src/wllama.ts
var LoggerWithoutDebug = __spreadProps(__spreadValues({}, console), {
  debug: () => {
  }
});
var WllamaError = class extends Error {
  constructor(message, type = "unknown_error") {
    super(message);
    __publicField(this, "type");
    this.type = type;
  }
};
var WllamaAbortError = class extends Error {
  constructor() {
    super("Operation aborted");
    __publicField(this, "name", "AbortError");
  }
};
var WllamaRuntimeError = class extends Error {
  constructor(message, stack) {
    super(message);
    __publicField(this, "name", "RuntimeError");
    __publicField(this, "stack");
    this.stack = stack;
  }
};
var Wllama = class {
  constructor(pathConfig, wllamaConfig = {}) {
    // The CacheManager and ModelManager are singleton, can be accessed by user
    __publicField(this, "cacheManager");
    __publicField(this, "modelManager");
    /** Feature detection: `loraBlobs` means `lora_adapters` accepts GGUF blobs, `nativeStates` means `native_states` loads natlang native state spans, `natlangScore` means `createNatlangScore` is available, `slotState` means `saveSlot` / `restoreSlot` / `eraseSlot` / `applyTemplate` are available and completions accept `id_slot`. */
    __publicField(this, "capabilities", {
      loraBlobs: true,
      nativeStates: true,
      natlangScore: true,
      slotState: true
    });
    __publicField(this, "slotFileId", 0);
    __publicField(this, "compat", null);
    __publicField(this, "proxy", null);
    __publicField(this, "config");
    __publicField(this, "pathConfig");
    __publicField(this, "useMultiThread", false);
    __publicField(this, "nbThreads", 1);
    __publicField(this, "useEmbeddings", false);
    __publicField(this, "useRerank", false);
    // available when loaded
    __publicField(this, "loadedContextInfo", null);
    __publicField(this, "seed");
    __publicField(this, "bosToken", -1);
    __publicField(this, "eosToken", -1);
    __publicField(this, "eotToken", -1);
    __publicField(this, "eogTokens", /* @__PURE__ */ new Set());
    __publicField(this, "addBosToken", false);
    __publicField(this, "addEosToken", false);
    __publicField(this, "mediaMarker");
    __publicField(this, "chatTemplate");
    __publicField(this, "metadata");
    __publicField(this, "hasEncoder", false);
    __publicField(this, "decoderStartToken", -1);
    // note: we overlay instead of using llama-server default_template_kwargs, because we cannot transfer complex data structure via GLUE
    // overlay allow mixed data type or nested structure for kwargs
    __publicField(this, "chatTemplateKwargs", {});
    var _a, _b, _c;
    checkEnvironmentCompatible();
    if (!pathConfig) throw new WllamaError("AssetsPathConfig is required");
    this.pathConfig = pathConfig;
    this.config = wllamaConfig;
    this.cacheManager = (_a = wllamaConfig.cacheManager) != null ? _a : new cache_manager_default();
    this.modelManager = (_c = wllamaConfig.modelManager) != null ? _c : new ModelManager({
      cacheManager: this.cacheManager,
      logger: (_b = wllamaConfig.logger) != null ? _b : console,
      parallelDownloads: wllamaConfig.parallelDownloads,
      allowOffline: wllamaConfig.allowOffline
    });
    this.setCompat("default");
  }
  logger() {
    var _a;
    return (_a = this.config.logger) != null ? _a : console;
  }
  checkModelLoaded() {
    if (!this.isModelLoaded()) {
      throw new WllamaError(
        "loadModel() is not yet called",
        "model_not_loaded"
      );
    }
  }
  /**
   * Get the libllama version string, e.g. "b6327-4d74393".
   *
   * @returns version string embedded at build time.
   */
  static getLibllamaVersion() {
    return LIBLLAMA_VERSION;
  }
  /**
   * Set compatibility options for Wllama.
   * @param compat Set to null to disable compatibility, or 'default' to use the default compat resources from CDN.
   * @param mode 'safari' by default; If set to 'firefox_safari', the compat mode will **also** be enabled on Firefox, which will significantly degrade the performance but allow using WebGPU on Firefox.
   */
  setCompat(compat, mode = "safari") {
    if (mode === "safari") {
      if (isFirefox()) {
        this.compat = null;
        return;
      }
    }
    this.compat = compat === "default" ? WasmCompatFromCDN : compat;
  }
  /**
   * Check if the model is loaded via `loadModel()`
   */
  isModelLoaded() {
    return !!this.proxy && !!this.metadata;
  }
  /**
   * Get token ID associated to BOS (begin of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getBOS() {
    return this.bosToken;
  }
  /**
   * Get token ID associated to EOS (end of sentence) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOS() {
    return this.eosToken;
  }
  /**
   * Get token ID associated to EOT (end of turn) token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getEOT() {
    return this.eotToken;
  }
  /**
   * Check if a given token is end-of-generation token (e.g. EOS, EOT, etc.)
   *
   * @param token the token ID to be checked
   * @returns true if the token is EOS, EOT, or any other end-of-generation tokens
   */
  isTokenEOG(token) {
    return token === this.eosToken || token === this.eotToken || this.eogTokens.has(token);
  }
  /**
   * Get token ID associated to token used by decoder, to start generating output sequence(only usable for encoder-decoder architecture). In other words, encoder uses normal BOS and decoder uses this token.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns -1 if the model is not loaded.
   */
  getDecoderStartToken() {
    return this.decoderStartToken;
  }
  /**
   * Get model hyper-parameters and metadata
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns ModelMetadata
   */
  getModelMetadata() {
    this.checkModelLoaded();
    return this.metadata;
  }
  /**
   * Check if we're currently using multi-thread build.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isMultithread() {
    this.checkModelLoaded();
    return this.useMultiThread;
  }
  /**
   * Get number of threads used in the current context.
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns number of threads
   */
  getNumThreads() {
    this.checkModelLoaded();
    return this.useMultiThread ? this.nbThreads : 1;
  }
  /**
   * Check if the current model uses encoder-decoder architecture
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if multi-thread is used.
   */
  isEncoderDecoderArchitecture() {
    this.checkModelLoaded();
    return this.hasEncoder;
  }
  /**
   * Must we add BOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if BOS token must be added to the sequence
   */
  mustAddBosToken() {
    this.checkModelLoaded();
    return this.addBosToken;
  }
  /**
   * Must we add EOS token to the tokenized sequence?
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns true if EOS token must be added to the sequence
   */
  mustAddEosToken() {
    this.checkModelLoaded();
    return this.addEosToken;
  }
  /**
   * Get the jinja chat template comes with the model. It only available if the original model (before converting to gguf) has the template in `tokenizer_config.json`
   *
   * NOTE: This can only being used after `loadModel` is called.
   *
   * @returns the jinja template. null if there is no template in gguf
   */
  getChatTemplate() {
    var _a;
    this.checkModelLoaded();
    return (_a = this.chatTemplate) != null ? _a : null;
  }
  /**
   * Check if WebGPU is supported by the current environment.
   * @returns true if WebGPU is supported
   */
  isSupportWebGPU() {
    return isSupportWebGPU();
  }
  /**
   * Load model from a given URL (or a list of URLs, in case the model is splitted into smaller files)
   * - If the model already been downloaded (via `downloadModel()`), then we will use the cached model
   * - Else, we download the model from internet
   * @param modelSourceOrURL
   * @param params
   */
  loadModelFromUrl(_0) {
    return __async(this, arguments, function* (modelSourceOrURL, params = {}) {
      var _a;
      const source = isString(modelSourceOrURL) ? { url: modelSourceOrURL } : modelSourceOrURL;
      const useCache = (_a = params.useCache) != null ? _a : true;
      const model = useCache ? yield this.modelManager.getModelOrDownload(source, params) : yield this.modelManager.downloadModel(source, params);
      const blobs = yield model.open();
      return yield this.loadModel(blobs, params);
    });
  }
  /**
   * Load model from a given Hugging Face model ID and file path.
   *
   * @param hfOptions
   * @param params
   */
  loadModelFromHF(_0) {
    return __async(this, arguments, function* (hfOptions, params = {}) {
      const source = yield getHFModelSource(hfOptions);
      return yield this.loadModelFromUrl(source, params);
    });
  }
  /**
   * Load model from a given list of Blob.
   *
   * You can pass multiple buffers into the function (in case the model contains multiple shards).
   *
   * @param ggufBlobsOrModel Can be either list of Blobs (in case you use local file), or a Model object (in case you use ModelManager)
   * @param params LoadModelParams
   */
  loadModel(_0) {
    return __async(this, arguments, function* (ggufBlobsOrModel, params = {}) {
      var _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k;
      const blobs = ggufBlobsOrModel instanceof Model ? yield ggufBlobsOrModel.open() : [...ggufBlobsOrModel];
      if (blobs.some((b) => b.size === 0)) {
        throw new WllamaError(
          "Input model (or splits) must be non-empty Blob or File",
          "load_error"
        );
      }
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (this.proxy) {
        throw new WllamaError("Module is already initialized", "load_error");
      }
      const supportMultiThread = yield isSupportMultiThread();
      const hwConccurency = Math.floor((navigator.hardwareConcurrency || 1) / 2);
      const nbThreads = (_a = params.n_threads) != null ? _a : hwConccurency;
      this.nbThreads = nbThreads;
      this.useMultiThread = supportMultiThread && nbThreads > 1;
      const workerResources = this.getWorkerResources();
      if (params.n_gpu_layers === 0) {
        workerResources.noWebGPU = true;
      }
      this.proxy = new ProxyToWorker(
        workerResources,
        this.useMultiThread ? nbThreads : 0,
        // 0 means disable pthread
        (_b = this.config.suppressNativeLog) != null ? _b : false,
        this.logger()
      );
      let logLevel = (_c = params.log_level) != null ? _c : 2 /* INFO */;
      if (this.config.suppressNativeLog) {
        logLevel = 9999;
      }
      const modelFiles = yield prepareBlobs(blobs);
      const loraPaths = [];
      const sideFiles = [];
      for (const [i, adapter] of ((_d = params.lora_adapters) != null ? _d : []).entries()) {
        if (adapter.blob) {
          if (adapter.blob.size === 0) {
            throw new WllamaError(
              `LoRA adapter ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `lora-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: adapter.blob });
          loraPaths.push(`models/${name}`);
        } else if (adapter.path) {
          loraPaths.push(adapter.path);
        } else {
          throw new WllamaError(
            `LoRA adapter ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      const statePaths = [];
      for (const [i, state] of ((_e = params.native_states) != null ? _e : []).entries()) {
        if (state.blob) {
          if (state.blob.size === 0) {
            throw new WllamaError(
              `Native state ${i} is an empty Blob`,
              "load_error"
            );
          }
          const name = `native-state-${padDigits(i + 1, 5)}.gguf`;
          sideFiles.push({ name, blob: state.blob });
          statePaths.push(`models/${name}`);
        } else if (state.path) {
          statePaths.push(state.path);
        } else {
          throw new WllamaError(
            `Native state ${i} needs a path or a blob`,
            "load_error"
          );
        }
      }
      yield this.proxy.moduleInit([...modelFiles.all, ...sideFiles]);
      this.logger().debug("Calling wllamaStart...");
      const startResult = yield this.proxy.wllamaStart();
      if (!startResult.success) {
        throw new WllamaError(
          `Error while calling start function, result = ${startResult}`
        );
      }
      this.logger().debug("Loading model...");
      const loadResult = yield this.proxy.wllamaAction("load", {
        _name: "load_req",
        log_level: logLevel,
        // if async read is not supported, use mmap; refer to README-dev.md for more details
        use_mmap: !canUseAsyncFileRead(workerResources.compat),
        use_mlock: false,
        n_gpu_layers: (_f = params.n_gpu_layers) != null ? _f : 99999,
        n_ctx: (_g = params.n_ctx) != null ? _g : 1024,
        n_threads: this.useMultiThread ? nbThreads : 1,
        n_ctx_auto: false,
        // not supported for now
        mmproj_path: modelFiles.mmproj ? `/models/${MMPROJ_FILE_NAME}` : void 0,
        model_paths: modelFiles.llm.map((f) => `models/${f.name}`),
        embeddings: params.embeddings,
        offload_kqv: params.offload_kqv,
        n_batch: params.n_batch,
        n_ubatch: params.n_ubatch,
        pooling_type: params.pooling_type,
        rope_scaling_type: params.rope_scaling_type,
        rope_freq_base: params.rope_freq_base,
        rope_freq_scale: params.rope_freq_scale,
        yarn_ext_factor: params.yarn_ext_factor,
        yarn_attn_factor: params.yarn_attn_factor,
        yarn_beta_fast: params.yarn_beta_fast,
        yarn_beta_slow: params.yarn_beta_slow,
        yarn_orig_ctx: params.yarn_orig_ctx,
        cache_type_k: params.cache_type_k,
        cache_type_v: params.cache_type_v,
        // with unified KV, all sequences share one n_ctx cache, so each request can still use the full context
        n_parallel: (_h = params.n_parallel) != null ? _h : 4,
        kv_unified: (_i = params.kv_unified) != null ? _i : true,
        flash_attn: params.flash_attn,
        swa_full: params.swa_full,
        chat_template: params.chat_template,
        jinja: params.jinja,
        reasoning: params.reasoning,
        image_min_tokens: params.image_min_tokens,
        image_max_tokens: params.image_max_tokens,
        warmup: params.warmup,
        no_kv_offload: params.no_kv_offload,
        mmproj_offload: params.mmproj_offload,
        cont_batching: params.cont_batching,
        n_keep: params.n_keep,
        ctx_shift: params.ctx_shift,
        cache_idle_slots: params.cache_idle_slots,
        n_cache_reuse: params.n_cache_reuse,
        lora_paths: params.lora_adapters ? loraPaths : void 0,
        lora_scales: (_j = params.lora_adapters) == null ? void 0 : _j.map((a) => {
          var _a2;
          return (_a2 = a.scale) != null ? _a2 : 1;
        }),
        lora_init_without_apply: params.lora_init_without_apply,
        native_state_paths: params.native_states ? statePaths : void 0,
        spec_draft_model: params.spec_draft_model,
        spec_draft_ngl: params.spec_draft_ngl,
        spec_draft_n_max: params.spec_draft_n_max,
        spec_draft_n_min: params.spec_draft_n_min,
        spec_draft_p_min: params.spec_draft_p_min,
        spec_draft_threads: params.spec_draft_threads,
        spec_draft_threads_batch: params.spec_draft_threads_batch,
        kv_overrides_keys: params.kv_overrides ? Object.keys(params.kv_overrides) : void 0,
        kv_overrides_vals: params.kv_overrides ? Object.values(params.kv_overrides) : void 0,
        reasoning_budget_tokens: params.reasoning_budget_tokens,
        reasoning_budget_message: params.reasoning_budget_message,
        reasoning_format: params.reasoning_format,
        skip_chat_parsing: params.skip_chat_parsing,
        prefill_assistant: params.prefill_assistant
      });
      const loadedCtxInfo = __spreadProps(__spreadValues({}, loadResult), {
        metadata: {}
      });
      for (let i = 0; i < loadResult.metadata_key.length; i++) {
        loadedCtxInfo.metadata[loadResult.metadata_key[i]] = loadResult.metadata_val[i];
      }
      this.seed = params.seed;
      this.bosToken = loadedCtxInfo.token_bos;
      this.eosToken = loadedCtxInfo.token_eos;
      this.eotToken = loadedCtxInfo.token_eot;
      this.useEmbeddings = !!params.embeddings;
      this.useRerank = params.pooling_type == "rank";
      this.metadata = {
        hparams: {
          nVocab: loadedCtxInfo.n_vocab,
          nCtxTrain: loadedCtxInfo.n_ctx_train,
          nEmbd: loadedCtxInfo.n_embd,
          nLayer: loadedCtxInfo.n_layer
        },
        meta: loadedCtxInfo.metadata
      };
      this.hasEncoder = !!loadedCtxInfo.has_encoder;
      this.decoderStartToken = loadedCtxInfo.token_decoder_start;
      this.addBosToken = loadedCtxInfo.add_bos_token;
      this.addEosToken = loadedCtxInfo.add_eos_token;
      this.chatTemplate = loadedCtxInfo.metadata["tokenizer.chat_template"];
      this.loadedContextInfo = loadedCtxInfo;
      this.eogTokens = new Set(loadedCtxInfo.list_tokens_eog);
      this.mediaMarker = loadedCtxInfo.media_marker;
      this.chatTemplateKwargs = (_k = params.default_template_kwargs) != null ? _k : {};
      this.logger().debug({ loadedCtxInfo });
    });
  }
  getLoadedContextInfo() {
    this.checkModelLoaded();
    if (!this.loadedContextInfo) {
      throw new WllamaError("Loaded context info is not available");
    }
    return __spreadValues({}, this.loadedContextInfo);
  }
  //////////////////////////////////////////////
  // High level API
  /**
   * Calculate embedding vector for a given text.
   * By default, BOS and EOS tokens will be added automatically. You can use the "skipBOS" and "skipEOS" option to disable it.
   * @param options OAI-compatible embedding creation options
   * @returns OAI-compatible embedding response
   */
  createEmbedding(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      if (!this.useEmbeddings) {
        throw new WllamaError(
          "Embeddings is not enabled. Please set it via LoadModelParams.embeddings"
        );
      }
      const result = yield this.proxy.wllamaAction(
        "embedding",
        {
          _name: "embd_req",
          data_json: JSON.stringify(options),
          files: []
          // TODO: support file input
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(options, false, result.req_id);
    });
  }
  /**
   * Rerank a list of documents against a query.
   * Requires the model to be loaded with embeddings: true and pooling_type: 'rank'.
   * @param options Reranking options (query, documents, top_n)
   * @returns Reranking response with relevance scores sorted highest first
   */
  createRerank(options) {
    return __async(this, null, function* () {
      var _a, _b;
      this.checkModelLoaded();
      if (!this.useEmbeddings || !this.useRerank) {
        throw new WllamaError(
          "Rerank is not enabled. Please set it via LoadModelParams: embeddings = true and pooling_type = rank"
        );
      }
      const top_n = (_a = options.top_n) != null ? _a : options.documents.length;
      let totalTokens = 0;
      const rawResults = [];
      for (let i = 0; i < options.documents.length; i++) {
        const result = yield this.proxy.wllamaAction("rerank", {
          _name: "rrnk_req",
          data_json: JSON.stringify({
            query: options.query,
            document: options.documents[i]
          })
        });
        if (!result.success) {
          throw new WllamaError(
            "Model failed to start reranking",
            "inference_error"
          );
        }
        const { score, tokens_evaluated } = yield this.getSingleResult(result.req_id, "reranking");
        totalTokens += tokens_evaluated;
        rawResults.push({ index: i, score });
      }
      rawResults.sort((a, b) => b.score - a.score);
      return {
        model: (_b = this.getModelMetadata().meta["general.name"]) != null ? _b : "",
        object: "list",
        usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
        results: rawResults.slice(0, top_n).map(({ index, score }) => ({
          index,
          relevance_score: score
        }))
      };
    });
  }
  /**
   * Answer typed questions about a state, using a decision model (TypeSafe-compatible System One API).
   * Image input is not supported yet.
   * @param options The state and the questions
   * @returns The answer of each question
   */
  createSystemOne(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "systemone",
        {
          _name: "sys1_req",
          data_json: JSON.stringify(options)
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start systemone",
          "inference_error"
        );
      }
      return yield this.getSingleResult(
        result.req_id,
        "systemone"
      );
    });
  }
  /**
   * Score candidate replies to chat messages (natlang decision readout): the log-probability of each reply as the closed final assistant message, summed over its tokens after the chat prompt. The replies of an item share the prompt.
   * @param options The items to score
   * @returns Per item, the log-probability and token count of each reply, or the error of an item that could not be rendered
   */
  createNatlangScore(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("score", {
        _name: "scor_req",
        data_json: JSON.stringify(options)
      });
      if (!result.success) {
        throw new WllamaError("Model failed to start scoring", "inference_error");
      }
      return yield this.getSingleResult(
        result.req_id,
        "score"
      );
    });
  }
  /**
   * Render chat messages (and tools) with the chat template, as the server's POST /apply-template does.
   * @param options OAI-compatible chat completion options (messages, tools, add_generation_prompt, ...)
   * @returns The prompt text
   */
  applyTemplate(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction(
        "apply_template",
        { _name: "tmpl_req", data_json: JSON.stringify(options) }
      );
      if (!result.success) {
        throw new WllamaError("Failed to apply chat template", "inference_error");
      }
      return result.prompt;
    });
  }
  /**
   * Save the state of a slot (its KV cache, tokens and context checkpoints), as the server's POST /slots/:id?action=save does. The slot must be idle.
   * @param idSlot The slot
   * @returns The saved state, to give to `restoreSlot` of an engine with the same model, build and context settings
   */
  saveSlot(idSlot) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const res = yield this.slotAction(
        idSlot,
        "save",
        filename
      );
      const state = yield this.proxy.slotFileRead(filename);
      return { state, n_saved: res.n_saved, n_written: res.n_written };
    });
  }
  /**
   * Restore a slot from a state of `saveSlot`, as the server's POST /slots/:id?action=restore does. A state that does not fit leaves the slot empty and throws.
   * @param idSlot The slot
   * @param state The saved state
   * @returns The number of restored tokens and bytes read
   */
  restoreSlot(idSlot, state) {
    return __async(this, null, function* () {
      const filename = `wllama-slot-${++this.slotFileId}.bin`;
      const data = state instanceof Uint8Array ? state : new Uint8Array(yield state.arrayBuffer());
      yield this.proxy.slotFileWrite(filename, data);
      try {
        return yield this.slotAction(idSlot, "restore", filename);
      } finally {
        yield this.proxy.slotFileRemove(filename);
      }
    });
  }
  /**
   * Clear the cached prompt of a slot, as the server's POST /slots/:id?action=erase does.
   * @param idSlot The slot
   */
  eraseSlot(idSlot) {
    return __async(this, null, function* () {
      return yield this.slotAction(idSlot, "erase", "");
    });
  }
  slotAction(idSlot, action, filename) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const result = yield this.proxy.wllamaAction("slots", {
        _name: "slot_req",
        id_slot: idSlot,
        action,
        filename
      });
      if (!result.success) {
        throw new WllamaError(`Failed to ${action} slot`, "inference_error");
      }
      return yield this.getSingleResult(result.req_id, `slot ${action}`);
    });
  }
  createChatCompletion(options) {
    return __async(this, null, function* () {
      var _a;
      if (Object.keys(this.chatTemplateKwargs).length > 0) {
        options = __spreadProps(__spreadValues({}, options), {
          chat_template_kwargs: __spreadValues(__spreadValues({}, this.chatTemplateKwargs), (_a = options.chat_template_kwargs) != null ? _a : {})
        });
      }
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  createCompletion(options) {
    return __async(this, null, function* () {
      if (options.stream && options.onData) {
        yield this.createCompletionImpl(options);
      } else if (options.stream) {
        return yield this.createCompletionGenerator(options);
      } else {
        return yield this.createCompletionImpl(__spreadProps(__spreadValues({}, options), { stream: false }));
      }
    });
  }
  /**
   * Private implementation of createCompletion
   */
  createCompletionImpl(options) {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      const isStream = !!options.stream;
      const isChat = !!options.messages;
      const customOpt = {};
      if (this.seed !== void 0) {
        customOpt.seed = this.seed;
      }
      let files = [];
      if (isChat) {
        const tmp = this.prepareMultimodalInput(
          options
        );
        options = tmp.params;
        files = tmp.files;
      }
      const result = yield this.proxy.wllamaAction(
        "completion",
        {
          _name: "cmpl_req",
          is_chat: isChat,
          data_json: JSON.stringify(__spreadValues(__spreadValues({}, options), customOpt)),
          files: files.map((f) => new Uint8Array(f))
        }
      );
      if (!result.success) {
        throw new WllamaError(
          "Model failed to start inference",
          "inference_error"
        );
      }
      return yield this.getResponse(
        options,
        isStream,
        result.req_id
      );
    });
  }
  /**
   * Same with `createCompletion`, but returns an async iterator instead.
   * Only called when stream=true and no onData is provided.
   */
  createCompletionGenerator(options) {
    return new Promise((resolve) => {
      const createGenerator = cbToAsyncIter(
        (callback) => {
          this.createCompletionImpl(__spreadProps(__spreadValues({}, options), {
            onData: (chunk) => callback(chunk)
          })).then(() => callback(void 0, true)).catch((err) => callback(void 0, false, err));
        }
      );
      resolve(createGenerator());
    });
  }
  /**
   * Whether the currently loaded model supports a specific input modality (e.g. image or audio).
   * @param modality
   * @returns
   */
  supportInputModality(modality) {
    this.checkModelLoaded();
    if (modality === "image") {
      return !!this.loadedContextInfo.has_image_input;
    } else if (modality === "audio") {
      return !!this.loadedContextInfo.has_audio_input;
    } else {
      throw new WllamaError(
        "Unsupported modality: " + modality,
        "unknown_error"
      );
    }
  }
  /**
   * Unload the model and free all memory.
   *
   * Note: This function will NOT crash if model is not yet loaded
   */
  exit() {
    return __async(this, null, function* () {
      var _a;
      yield (_a = this.proxy) == null ? void 0 : _a.wllamaExit();
      this.proxy = null;
    });
  }
  /**
   * [FOR DEBUGGING ONLY] Run ggml backend ops tests without loading any model.
   *
   * Initializes the wasm runtime, executes `test-backend-ops` with the given args, then shuts down.
   *
   * For more info, please refer to guides/debug.md
   *
   * @param args Arguments forwarded to test-backend-ops (e.g. ["-o", "ADD"])
   * @returns retcode (0 = all tests passed) and success flag
   */
  testBackendOps() {
    return __async(this, arguments, function* (args = []) {
      var _a;
      if (!this.pathConfig["default"]) {
        throw new WllamaError(
          '"default" is missing from pathConfig',
          "load_error"
        );
      }
      if (!(yield isSupportMultiThread())) {
        throw new WllamaError(
          "Multi-threading is required to run backend ops tests, but it is not supported in the current environment."
        );
      }
      const tmpProxy = new ProxyToWorker(
        this.getWorkerResources(),
        0,
        // single-thread; no model needed
        (_a = this.config.suppressNativeLog) != null ? _a : false,
        this.logger()
      );
      try {
        yield tmpProxy.moduleInit([]);
        const startResult = yield tmpProxy.wllamaStart();
        if (!startResult.success) {
          throw new WllamaError(
            `Error while calling start function, result = ${startResult}`
          );
        }
        const result = yield tmpProxy.wllamaAction(
          "test_backend_ops",
          { _name: "tbop_req", args: ["test-backend-ops", ...args] }
        );
        return { retcode: result.retcode, success: result.success };
      } finally {
        yield tmpProxy.wllamaExit();
      }
    });
  }
  //////////////////////////////////////////////
  // Low level API
  // TODO: add back
  /**
   * get debug info
   */
  _getDebugInfo() {
    return __async(this, null, function* () {
      this.checkModelLoaded();
      return yield this.proxy.wllamaDebug();
    });
  }
  //////////////////////////////////////////////
  // Utils
  jsonDecode(data_json) {
    try {
      return JSON.parse(data_json);
    } catch (e) {
      this.logger().error("Failed to parse JSON:", data_json);
      throw new WllamaError("Failed to parse model output", "inference_error");
    }
  }
  prepareMultimodalInput(params) {
    const msg = params.messages;
    const msgNew = [];
    const files = [];
    for (const m of msg) {
      if (Array.isArray(m.content)) {
        const newContent = [];
        for (const c of m.content) {
          if (c.type === "text" || c.type === "native_state") {
            newContent.push(c);
          } else {
            if (!this.mediaMarker) {
              throw new WllamaError(
                "Media marker is undefined",
                "inference_error"
              );
            }
            files.push(c.data);
            newContent.push({
              type: "text",
              text: this.mediaMarker
            });
          }
        }
        msgNew.push(__spreadProps(__spreadValues({}, m), {
          content: newContent
        }));
      } else {
        msgNew.push(m);
      }
    }
    return {
      params: __spreadProps(__spreadValues({}, params), {
        messages: msgNew
      }),
      files
    };
  }
  // release the slot occupied by the request; cancelling an already-finished request is a no-op
  cancelRequest(reqId) {
    return __async(this, null, function* () {
      try {
        yield this.proxy.wllamaAction("cancel", {
          _name: "cncl_req",
          req_id: reqId
        });
      } catch (e) {
        this.logger().warn("Failed to cancel request", reqId, e);
      }
    });
  }
  getSingleResult(reqId, task) {
    return __async(this, null, function* () {
      let completed = false;
      try {
        while (true) {
          const chunk = yield this.proxy.wllamaAction(
            "get_result",
            { _name: "gres_req", req_id: reqId }
          );
          const jsonString = chunk.data_json;
          if (jsonString && jsonString.length > 0) {
            if (chunk.is_error) {
              const jsonData = this.jsonDecode(jsonString);
              throw new WllamaError(
                jsonData.message || `Unknown ${task} error`,
                "inference_error"
              );
            }
            completed = true;
            return this.jsonDecode(jsonString);
          }
          if (!chunk.has_more) {
            completed = true;
            break;
          }
        }
        throw new WllamaError(`No ${task} result received`, "inference_error");
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
    });
  }
  getResponse(options, isStream, reqId) {
    return __async(this, null, function* () {
      var _a, _b;
      let finalResult = null;
      let completed = false;
      try {
        while (true) {
          if ((_a = options.abortSignal) == null ? void 0 : _a.aborted) {
            throw new WllamaAbortError();
          }
          const result_chunk = yield this.proxy.wllamaAction(
            "get_result",
            {
              _name: "gres_req",
              req_id: reqId
            }
          );
          const jsonString = result_chunk.data_json;
          if (!jsonString || jsonString.length === 0) {
            if (!result_chunk.has_more) {
              completed = true;
              break;
            } else {
              continue;
            }
          }
          if (jsonString == "null") {
            continue;
          }
          let jsonData = this.jsonDecode(jsonString);
          finalResult = jsonData;
          if (result_chunk.is_error) {
            this.logger().error("Model returned an error:", jsonData);
            throw new WllamaError(
              jsonData.message || "Unknown inference error",
              "inference_error"
            );
          }
          if (isStream) {
            if (!Array.isArray(jsonData)) {
              jsonData = [jsonData];
            }
            for (const chunk of jsonData) {
              (_b = options.onData) == null ? void 0 : _b.call(options, chunk);
              finalResult = chunk;
            }
          }
          if (!result_chunk.has_more) {
            completed = true;
            break;
          }
        }
      } finally {
        if (!completed) {
          yield this.cancelRequest(reqId);
        }
      }
      return finalResult;
    });
  }
  getWorkerResources() {
    const workerResources = {
      wasmPath: absoluteUrl(this.pathConfig["default"]),
      compat: false
    };
    if (needCompat()) {
      if (!this.compat) {
        this.logger().warn(
          "Not using compat mode" + (isFirefox() ? " (expected on Firefox - WebGPU will be disabled)" : "")
        );
      } else {
        const isUsingDefault = this.compat.worker === WasmCompatFromCDN.worker && this.compat.wasm === WasmCompatFromCDN.wasm;
        if (isUsingDefault) {
          this.logger().warn(
            "Compatibility mode is activated, using resources from CDN. To use local resources, please refer to @wllama/wllama-compat package."
          );
          this.logger().warn(
            "IMPORTANT: Performance will be significantly degraded in compatibility mode."
          );
        }
        workerResources.wasmPath = absoluteUrl(this.compat.wasm);
        workerResources.jsPath = this.compat.worker;
        workerResources.compat = true;
      }
    }
    if (isFirefox()) {
      if (workerResources.compat) {
        this.logger().warn(
          'Using compat mode on Firefox, performance will be significantly degraded; Consider enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      } else if (!isSupportJSPI()) {
        this.logger().warn(
          'WebGPU is disabled on Firefox due to missing JSPI support. Please consider enabling compat mode, or enabling "javascript.options.wasm_js_promise_integration" in "about:config".'
        );
      }
    }
    return workerResources;
  }
};
export {
  CacheManager,
  LogLevel,
  LoggerWithoutDebug,
  Model,
  ModelManager,
  ModelValidationStatus,
  POLYFILL_ETAG,
  Wllama,
  WllamaAbortError,
  WllamaError,
  WllamaRuntimeError,
  getHFFileSHA256,
  getHFModelSource,
  isValidGgufFile
};
